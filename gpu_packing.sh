#!/bin/sh

#SBATCH --ntasks-per-node=3
#SBATCH --ntasks=3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=18G
#SBATCH --time=4:00:00
srun -l --output=slurm-%j-%t.out --multi-prog gpu_packing.conf
