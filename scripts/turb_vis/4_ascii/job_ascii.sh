#!/bin/bash -l
#
#
# 8 nodes, 8 MPI task per node
#
#SBATCH --job-name="extract"
#SBATCH --time=3:00:00
#SBATCH --nodes=9
#SBATCH --ntasks=90
#========================================


# Only using 1 node for now
# Mainly because I'm not really in a hurry
# for 90 partitions, it takes: 02:18:03
# If using 90 ranks (9 nodes, 10 ranks per node) it takes ~20min
module load h5py

mpiexec -n 90 python ascii.py /users/staff/uniadm/zhu0002/extract/slice /storage/shared/projects/sph-exa/data/slice 0 parallel

# If parallel
# srun --cpu_bind=sockets python extract.py /scratch/snx3000/sebkelle/chkp.avc.final.h5 0 parallel