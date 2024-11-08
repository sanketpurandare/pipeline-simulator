import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.distributed.pipelining import ScheduleGPipe, Schedule1F1B
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining import pipeline, SplitPoint


in_dim = 512
layer_dims = [512, 1024, 256]
out_dim = 10

# Single layer definition
class MyNetworkBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.lin(x)
        x = torch.relu(x)
        return x


# Full model definition
class MyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_layers = len(layer_dims)

        prev_dim = in_dim
        # Add layers one by one
        for i, dim in enumerate(layer_dims):
            super().add_module(f"layer{i}", MyNetworkBlock(prev_dim, dim))
            prev_dim = dim

        # Final output layer (with OUT_DIM projection classes)
        self.output_proj = torch.nn.Linear(layer_dims[-1], out_dim)

    def forward(self, x):
        for i in range(self.num_layers):
            layer = getattr(self, f"layer{i}")
            x = layer(x)

        return self.output_proj(x)


def run_test(gpu_id, world_size):
    # print(dist.get_rank())
    # print("Initialized Rank: ", gpu_id)
    # print("Device Count: ", torch.cuda.device_count())
    # print("Device id", torch.cuda.current_device())

    rank = gpu_id
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    else:
        device = torch.device("cpu")
    stage_idx = rank

    batch_size = 32
    example_input = torch.randn(batch_size, in_dim, device=device)
    n_microbatches = 4

    mod = MyNetwork().to(device)

    example_input = torch.randn(int(batch_size / n_microbatches), in_dim, device=device)
    pipe = pipeline(
        module=mod,
        mb_args=(example_input,),
        split_spec={
            "layer0": SplitPoint.END,
            "layer1": SplitPoint.END,        
        }
    )

    stage = pipe.build_stage(stage_idx, device)

    # Create a schedule
    schedule = ScheduleGPipe(stage, n_microbatches)

    # Input data (whole batch)
    x = torch.randn(batch_size, in_dim, device=device)

    # Run the pipeline with input `x`
    # `x` will be divided into microbatches automatically
    if rank == 0:
        schedule.step(x)
    else:
        output = schedule.step()


if __name__ == "__main__":
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "29500"
    # world_size = 2
    # mp.spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)

    try:
        dist.init_process_group(backend="nccl")
        gpu_id = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        run_test(gpu_id, world_size)
        dist.destroy_process_group()
    except Exception as e:
        print(str(e))
        print("Unsuccessful.")