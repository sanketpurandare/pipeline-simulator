import os
import torch.multiprocessing as mp
import torch.distributed as dist
import torch
from torch.distributed.pipelining import ScheduleGPipe
from torch.distributed.pipelining import PipelineStage



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


# Model chunk definition
class ModelChunk0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer0 = MyNetworkBlock(in_dim, layer_dims[0])

    def forward(self, x):
        return self.layer0(x)

class ModelChunk1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = MyNetworkBlock(layer_dims[0], layer_dims[1])

    def forward(self, x):
        return self.layer1(x)

class ModelChunk2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer2 = MyNetworkBlock(layer_dims[1], layer_dims[2])
        # Final output layer (with OUT_DIM projection classes)
        self.output_proj = torch.nn.Linear(layer_dims[2], out_dim)

    def forward(self, x):
        x = self.layer2(x)
        return self.output_proj(x)


def run_worker(rank: int, world_size: int):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print("Initialized Rank: ", rank)
    print("Device Count: ", torch.cuda.device_count())
    print("Device id", torch.cuda.current_device())
    dist.destroy_process_group()

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

    # Create the model chunks
    batch_size = 32
    n_microbatches = 4
    example_input_stage_0 = torch.randn(int(batch_size / n_microbatches), in_dim, device=device)
    example_input_stage_1 = torch.randn(int(batch_size / n_microbatches), layer_dims[0], device=device)
    example_input_stage_2 = torch.randn(int(batch_size / n_microbatches), layer_dims[1], device=device)

    rank_model_and_input = {
        0: (ModelChunk0(), example_input_stage_0),
        1: (ModelChunk1(), example_input_stage_1),
        2: (ModelChunk2(), example_input_stage_2),
    }

    # Pipeline stage is our main pipeline runtime. It takes in the pipe object,
    # the rank of this process, and the device.
    if rank in rank_model_and_input:
        model, example_input = rank_model_and_input[rank]
        model.to(device)  # Move the model to the device
        stage = PipelineStage(
            submodule=model,
            stage_index=rank,
            num_stages=world_size,
            device=device,
            input_args=example_input,
        )
        print(f"Rank {rank} initialized")
    else:
        raise RuntimeError("Invalid rank")

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
        print(e)
        print("Unsuccessful.")