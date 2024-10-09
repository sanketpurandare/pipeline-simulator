#!/bin/bash

#SBATCH --job-name=pp-sim
#SBATCH --partition=seas_gpu,gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:2
#SBATCH --time=1:00:00
#SBATCH --mem=64gb
#SBATCH --output=/n/holyscratch01/idreos_lab/Users/spurandare/pipeline-simulator/job_logs/%x_%j.out
#SBATCH --error=/n/holyscratch01/idreos_lab/Users/spurandare/pipeline-simulator/job_logs/%x_%j.err
#SBATCH --open-mode=append
#SBATCH --chdir=/n/holyscratch01/idreos_lab/Users/spurandare/pipeline-simulator

scontrol show job $SLURM_JOB_ID

head_node=$(echo "$SLURM_NODELIST" | cut -d ',' -f 1)
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo Node IP: $head_node_ip

export NCCL_DEBUG=INFO

GPUS_PER_NODE=$(scontrol show job $SLURM_JOB_ID | grep "gres/gpu=" | head -1 | awk -F= '{print $NF}')
echo "GPUS_PER_NODE: $GPUS_PER_NODE"
echo "SLURM_NNODES: $SLURM_NNODES"

srun --output job_logs/%x_%j_%t.out --error job_logs/%x_%j_%t.err \
  torchrun --nnodes ${SLURM_NNODES} --nproc_per_node ${GPUS_PER_NODE} \
  --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "$head_node_ip:29500" \
  test_multi-gpu.py