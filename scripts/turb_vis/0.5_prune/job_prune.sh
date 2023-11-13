#!/bin/bash -l
#SBATCH --job-name="prune"
#SBATCH --time=9:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=gpu
#SBATCH --account=c32
#SBATCH --mail-user="yiqing.zhu@unibas.ch"
#SBATCH --mail-type=ALL
#========================================


# Only using 1 node for now
# Mainly because I'm not really in a hurry
module load daint-gpu
module load h5py

srun --cpu_bind=sockets python prune.py /scratch/snx3000/yzhu/extract_vx/res/slice00000.h5 /scratch/snx3000/yzhu/prune/res/res_800.h5 800