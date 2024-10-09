# To run this script on a single node with two GPUs in an interactive way:
# 1. Connect to a 1 node with 2 GPUs: srun -p gpu,seas_gpu -t 0-1:00 --mem 256000 --gres=gpu:2 -c 2 --pty bash
# 2. Run this command: torchrun --standalone --nproc_per_node=2 test_multi-gpu.py 

# To run this script on multiple nodes with multiple GPUs in a non-interactive way:
# 1. Create a folder job_logs
# 2. chmod +x run_distributed.sbatch
# 3. sbatch run_distributed.sbatch (This will by default run with 1 node and 2 GPUs)
# 4. sbatch --nodes=x --gres=gpu:nvidia_a100-sxm4-80gb:y run_distributed.sbatch 
# (This will override the number of nodes to be x and number of gpus per node to be y)

import os
import torch.distributed as dist
import torch


def run_test(gpu_id: int, world_size: int):
    print("Distributed Rank", dist.get_rank())
    print("Initialized Rank: ", gpu_id)
    print("Device Count: ", torch.cuda.device_count())
    print("Device id", torch.cuda.current_device())
    dist.barrier()


if __name__ == "__main__":

    try:
        dist.init_process_group(backend="nccl")
        gpu_id = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        run_test(gpu_id, world_size)
        dist.destroy_process_group()
    except Exception:
        print("Unsuccessful.")