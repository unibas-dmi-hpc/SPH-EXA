#!/bin/bash -l
#
#
# 8 nodes, 8 MPI task per node
#
#SBATCH --job-name="extract"
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=gpu
#SBATCH --account=c32
#========================================


# Only using 1 node for now
# Mainly because I'm not really in a hurry
# for 90 partitions, it takes: 02:18:03
# If using 90 ranks (9 nodes, 10 ranks per node) it takes ~20min
module load daint-gpu
module load h5py

srun --cpu_bind=sockets python extract.py /scratch/snx3000/sebkelle/chkp.avc.final.h5 /scratch/snx3000/yzhu/extract/res/slice 0 serial

# If parallel
# srun --cpu_bind=sockets python extract.py /scratch/snx3000/sebkelle/chkp.avc.final.h5 0 parallel