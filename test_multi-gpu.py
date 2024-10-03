import os
import torch.multiprocessing as mp
import torch.distributed as dist
import torch

def run_worker(rank: int, world_size: int):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print("Initialized Rank: ", rank)
    print("Device Count: ", torch.cuda.device_count())
    print("Device id", torch.cuda.current_device())
    dist.destroy_process_group()

def run_test(gpu_id, world_size):
    print(dist.get_rank())
    print("Initialized Rank: ", gpu_id)
    print("Device Count: ", torch.cuda.device_count())
    print("Device id", torch.cuda.current_device())




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
    except Exception:
        print("Unsuccessful.")