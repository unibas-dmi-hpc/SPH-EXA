#!/bin/bash -l
#
#
# 1 nodes, 1 MPI task per node
#
#SBATCH --job-name="merge"
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --constraint=gpu
#SBATCH --account=c32
#========================================


# Normally you don't need to be parallel to merge the files
# Numpy data reading takes 6-10s for one file.
module load daint-gpu

srun --cpu_bind=sockets python merge.py /scratch/snx3000/yzhu/interpolate/res/interpolated_ /scratch/snx3000/yzhu/merge/res/merged_ 90