#!/bin/bash -l
#
# 9 nodes, 10 MPI task per node
# For a 4GB file (300,000,000 particles), takes ~3h
#
#SBATCH --job-name="interpolate"
#SBATCH --time=10:00:00
#SBATCH --nodes=9
#SBATCH --ntasks-per-node=10
#SBATCH --ntasks=90
#SBATCH --constraint=gpu
#SBATCH --account=c32
#========================================
# load modules
module load daint-gpu
module load h5py

srun --cpu_bind=sockets python interpolate.py parallel /scratch/snx3000/yzhu/extract/res/slice 0 /scratch/snx3000/yzhu/interpolate/res/interpolated_
