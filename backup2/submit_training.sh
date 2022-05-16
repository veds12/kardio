#!/bin/bash
#SBATCH --gres=gpu:rtx8000:1     # Number of GPUs (per node)
#SBATCH --mem=8G        # memory (per node)
#SBATCH --time=01-05:00:00   # time (DD-HH:MM)

### cluster information above this line

### load environment
module load anaconda/3
module load cuda/10.1
conda activate kardio

### run script
CUDA_LAUNCH_BLOCKING=1 python train.py --seed 45 --verbose --epochs 30 --logging --name Kardio_New_Pipeline_0.3_long --gt_coeff 0.3 --checkpoint ./checkpoints --epochs 300