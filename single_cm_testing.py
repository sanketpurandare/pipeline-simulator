import os
import sys
import torch.multiprocessing as mp
import torch.distributed as dist
import torch
import torch.nn as nn
import torch.nn.functional as F

from llama2_model import Transformer, ModelArgs

from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._tools.mem_tracker import MemTracker
from torch.distributed._tools.runtime_estimator import RuntimeEstimator
from contextlib import nullcontext
from torch.testing._internal.distributed.fake_pg import FakeStore

from torch import optim
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe, Schedule1F1B
from torch.distributed.pipelining import pipeline, SplitPoint
from torch.distributed._tools.fake_collectives import *
from itertools import chain
from example_context_manager import capture_collectives

_world_size = 8
vocab_size = 32000

def loss_fn(outputs, labels):
    return F.cross_entropy(outputs, labels)

def split_model(model, rank):
    if rank == 0:
        model.norm = None
        model.output = None
    elif rank == _world_size - 1:
        model.tok_embeddings = None
    else:
        model.norm = None
        model.tok_embeddings = None
        model.output = None
    return model

def subprocess(gpu_id, world_size):
    # dev = torch.device("cuda:0")
    # torch.cuda.set_device(dev)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(gpu_id)
    rank = gpu_id
    store = FakeStore()
    torch.distributed.init_process_group(
        "fake", rank=rank, world_size=_world_size, store=store
    )
    
    n_layers = 16
    n_microbatches = 8
    batch_size = 8
    seq_length = 512
    num_stages = _world_size

    # create model and move it to GPU - init"cuda"_mesh has already mapped GPU ids.
    simple_llama2_config = ModelArgs(dim=2048, n_layers=int(n_layers/n_microbatches), n_heads=32, vocab_size=32000)
    fake_mode = True
    with FakeTensorMode() if fake_mode else nullcontext():

        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        else:
            device = torch.device("cpu")

        with torch.device(device):
            model = Transformer.from_model_args(simple_llama2_config)
            
            sub_model = split_model(model, rank)

            if rank == 0:
                example_input = torch.randint(0, model.vocab_size, (int(batch_size / n_microbatches), seq_length), device=device)
                example_output = torch.randn(int(batch_size / n_microbatches), seq_length, model.model_dim, device=device)
            elif rank == world_size - 1:
                example_input = torch.randn(int(batch_size / n_microbatches), seq_length, model.model_dim, device=device)
                example_output = torch.randn(int(batch_size / n_microbatches), seq_length, model.vocab_size, device=device)
            else:
                example_input = torch.randn(int(batch_size / n_microbatches), seq_length, model.model_dim, device=device)
                example_output = torch.randn(int(batch_size / n_microbatches), seq_length, model.model_dim, device=device)

            stage = PipelineStage(
                submodule=sub_model,
                stage_index=rank,
                num_stages=world_size,
                device=device,
                input_args=example_input,
                output_args=example_output,
            )

        optim = torch.optim.Adam(model.parameters(), foreach=True)
        mem_tracker = MemTracker()
        mem_tracker.track_external(model, optim)
        runtime_estimator = RuntimeEstimator(rank)
        
        # Create a schedule
        schedule = Schedule1F1B(stage, n_microbatches, loss_fn=loss_fn)
        
        # Input data (whole batch)
        x = torch.randint(0, model.vocab_size, (batch_size, seq_length), device=device)
        target = torch.randn(batch_size, seq_length, model.model_dim, device=device)
        if rank == world_size - 1:
            target = torch.randn(batch_size, seq_length, model.vocab_size, device=device)

        # Run the pipeline with input `x`
        # `x` will be divided into microbatches automatically
        if rank == 0:
            with capture_collectives():
                with runtime_estimator("operator-level-benchmark"):
                    with mem_tracker as mt:
                        schedule.step(x, target=target)
                        mt.display_modulewise_snapshots(depth=1, units="MiB", tabulate=True)
                    runtime_estimator.display_modulewise_stats(depth=1)
                
        else:
            with capture_collectives():
                with runtime_estimator("operator-level-benchmark"):
                    with mem_tracker as mt:
                        output = schedule.step(target=target)
                        mt.display_modulewise_snapshots(depth=1, units="MiB", tabulate=True)
                    runtime_estimator.display_modulewise_stats(depth=1)

        # # Training loop:
        # # Perform a num of iterations of forward/backward
        # # and optimizations for the sharded module.
        # print("\nStarting 2D training...")
        # num_iterations = 2
        # batch_size = 2
        # torch.cuda.reset_accumulated_memory_stats()
        # torch.cuda.reset_peak_memory_stats()
        # mem_tracker = MemTracker()
        # mem_tracker.track_external(sharded_model, optimizer)
        # for i in range(num_iterations):
        #     # seeding with dp_rank to ensure identical inputs for TP groups
        #     with mem_tracker:
        #         torch.manual_seed(i + dp_rank)
        #         inp = torch.randint(32000, (8, 512), device=dev)

        #         output = sharded_model(inp)
        #         output.sum().backward()
        #         optimizer.step()
        #     if i == 0:
        #         mem_tracker.reset_mod_stats()

        # mem_tracker.display_snapshot("peak", units="GiB", tabulate=True)
        # mem_stats = torch.cuda.memory_stats()
        # peak_active = mem_stats["active_bytes.all.peak"]
        # peak_reserved = mem_stats["reserved_bytes.all.peak"]

        # print(
        #     f"peak active: {peak_active / gib:.2f} GiB | "
        #     f"peak reserved: {peak_reserved / gib:.2f} GiB"
        # )

if __name__ == "__main__":
    try: 
        mp.spawn(subprocess, args=(_world_size,), nprocs=_world_size, join=True)
    except Exception as e:
        print(e)
        print("Unsuccessful.")