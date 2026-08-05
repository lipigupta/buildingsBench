#!/bin/bash
# Runs train_model.py across multiple nodes/GPUs in ONE job using srun

#SBATCH -A m4388_g                  
#SBATCH -C gpu
#SBATCH -q debug                # debug while testing
#SBATCH -N 2                      # 4 nodes x 4 GPUs/node = 16 GPUs total -- change -N to scale
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4       # one task per GPU
#SBATCH --gpus-per-task=1         # each task is bound to exactly 1 physical GPU
#SBATCH -c 32
#SBATCH -t 00:10:00                
#SBATCH -J train-model-parallel
#SBATCH -o train_model_%j.out
#SBATCH -e train_model_%j.err

module load python                 # TODO: match your usual module load
conda activate buildingsEnv      # TODO: your conda env name

export REPO_PATH="/pscratch/sd/l/lgupta/buildingsBench/"
export BUILDINGS_BENCH="/global/cfs/cdirs/m4388/2025_Bootcamp/Project4/Dataset"
export TRANSFORM_PATH="/global/cfs/cdirs/m4388/2025_Bootcamp/Project4/Dataset/metadata/transforms"

# One srun launches all 16 tasks (4 nodes x 4 tasks/node) at once. Slurm sets
# SLURM_PROCID (0-15, this task's global rank) and SLURM_NTASKS (16) for each
# task automatically -- train_model.py reads those directly, so no wrapper
# script or manual rank math is needed here.
srun python Train-Model-EP.py --task both
