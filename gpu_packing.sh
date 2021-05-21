#!/bin/sh

#SBATCH --ntasks-per-node=3
#SBATCH --ntasks=3
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=18G
#SBATCh --time=3:00:00
srun -l --output=slurm-%j-%t.out --multi-prog gpu_packing.conf
