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
from torch.distributed.pipelining import PipelineStage, ScheduleInterleaved1F1B, ScheduleLoopedBFS, ScheduleInterleavedZeroBubble, ScheduleZBVZeroBubble
from torch.distributed._tools.fake_collectives import *
from itertools import chain
from example_context_manager import capture_collectives

# Number of GPUs
_world_size = 8

def loss_fn(outputs, labels):
    return F.cross_entropy(outputs, labels)

# Only have input layer on first stage and output layer on last stage
def trim_model(model, rank, i, num_stages):
    if rank == 0 and i == 0:
        model.norm = None
        model.output = None
    elif rank == _world_size - 1 and i == num_stages - 1:
        model.tok_embeddings = None
    else:
        model.norm = None
        model.tok_embeddings = None
        model.output = None
    return model

def subprocess(gpu_id, world_size):
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(gpu_id)
    rank = gpu_id
    store = FakeStore()
    torch.distributed.init_process_group(
        "fake", rank=rank, world_size=_world_size, store=store
    )
    
    # Llama parameters
    vocab_size = 32000
    n_heads = 32
    dim = 2048
    batch_size = 8
    seq_length = 512

    # Pipeline parameters
    n_layers = 16
    n_microbatches = 8
    num_stages_per_GPU = 2
    trackers_on = True

    # create appropriate model stages
    simple_llama2_config = ModelArgs(dim=dim, n_layers=int(n_layers/(num_stages_per_GPU*n_microbatches)), n_heads=n_heads, vocab_size=vocab_size)
    fake_mode = True
    with FakeTensorMode() if fake_mode else nullcontext():

        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        else:
            device = torch.device("cpu")
            
        stages = []

        with torch.device(device):
            for i in range(num_stages_per_GPU):
                model = Transformer.from_model_args(simple_llama2_config)
                
                sub_model = trim_model(model, rank, i, num_stages_per_GPU)

                if rank == 0 and i == 0: # first stage for GPU 0 contains first layer of model
                    example_input = torch.randint(0, model.vocab_size, (int(batch_size / n_microbatches), seq_length), device=device)
                    example_output = torch.randn(int(batch_size / n_microbatches), seq_length, model.model_dim, device=device)
                elif rank == world_size - 1 and i == num_stages_per_GPU - 1: # last stage for GPU n contains last layer of model
                    example_input = torch.randn(int(batch_size / n_microbatches), seq_length, model.model_dim, device=device)
                    example_output = torch.randn(int(batch_size / n_microbatches), seq_length, model.vocab_size, device=device)
                else:
                    example_input = torch.randn(int(batch_size / n_microbatches), seq_length, model.model_dim, device=device)
                    example_output = torch.randn(int(batch_size / n_microbatches), seq_length, model.model_dim, device=device)

                stage = PipelineStage(
                    submodule=sub_model,
                    stage_index=rank + i * _world_size,
                    num_stages=num_stages_per_GPU * _world_size,
                    device=device,
                    input_args=example_input,
                    output_args=example_output,
                )
                
                stages.append(stage)
            
        optim = torch.optim.Adam(model.parameters(), foreach=True)
        mem_tracker = MemTracker()
        mem_tracker.track_external(model, optim)
        runtime_estimator = RuntimeEstimator(rank)
        
        # Create a schedule
        schedule = ScheduleInterleaved1F1B(stages, n_microbatches, loss_fn=loss_fn)
        
        # Input data (whole batch)
        x = torch.randint(0, model.vocab_size, (batch_size, seq_length), device=device)
        if rank == world_size - 1:
            target = torch.randn(batch_size, seq_length, model.vocab_size, device=device)
        else:
            target = torch.randn(batch_size, seq_length, model.model_dim, device=device)

        # Run the pipeline with input `x`
        # `x` will be divided into microbatches automatically
        if rank == 0:
            if trackers_on:
                with capture_collectives():
                    with runtime_estimator("operator-level-benchmark"):
                        with mem_tracker as mt:
                            schedule.step(x, target=target)
                            mt.display_modulewise_snapshots(depth=1, units="MiB", tabulate=True)
                        runtime_estimator.display_modulewise_stats(depth=1)
            else:
                schedule.step(x, target=target)
                
        else:
            if trackers_on:
                with capture_collectives():
                    with runtime_estimator("operator-level-benchmark"):
                        with mem_tracker as mt:
                            output = schedule.step(target=target)
                            mt.display_modulewise_snapshots(depth=1, units="MiB", tabulate=True)
                        runtime_estimator.display_modulewise_stats(depth=1)
            else:
                output = schedule.step(target=target)

if __name__ == "__main__":
    try: 
        mp.spawn(subprocess, args=(_world_size,), nprocs=_world_size, join=True)
    except Exception as e:
        print(e)
        print("Unsuccessful.")