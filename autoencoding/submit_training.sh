#!/bin/bash
#SBATCH --gres=gpu:1     # Number of GPUs (per node)
#SBATCH --mem=8G        # memory (per node)
#SBATCH --time=01-05:00:00   # time (DD-HH:MM)
#SBATCH --output=/network/scratch/v/vedant.shah/kardio/slurms/kardio-ae-%j.out

### cluster information above this line

### load environment
module load anaconda/3
module load cuda/10.1
conda activate kardio

### arguments
model_name=$1

### run script
CUDA_LAUNCH_BLOCKING=1 python train_ae.py --verbose --logging --model_name ${model_name} --epochs 200 --name ${model_name}-ae --checkpoint ./checkpoints --test_every 3 --chkpt_every 5