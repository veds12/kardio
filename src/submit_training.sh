#!/bin/bash
#SBATCH --gres=gpu:1     # Number of GPUs (per node)
#SBATCH --mem=8G        # memory (per node)
#SBATCH --time=01-05:00:00   # time (DD-HH:MM)
#SBATCH --output=/network/scratch/v/vedant.shah/slurms/kardio-cnn-%j.out

### cluster information above this line

### load environment
module load anaconda/3
module load cuda/10.1
conda activate li-rarl

seed=$1
model=$2
name=$3


### run training script
CUDA_LAUNCH_BLOCKING=1 python train.py --seed ${seed} --model ${model} --logging 1 --verbose 1 --name ${name} --checkpoint ./checkpoint --epochs 100
