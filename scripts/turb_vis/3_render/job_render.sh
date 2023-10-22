#!/bin/bash -l
#
#
# 8 nodes, 8 MPI task per node
#
#SBATCH --job-name="render"
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --constraint=gpu
#SBATCH --account=c32
#========================================


# To run this you need a python environment with:
# - Pillow
# - matplotlib
# - numpy
# No need for parallellization. It's fast enough (on a proper vis GPU)
module load daint-gpu

srun --cpu_bind=sockets python render.py /scratch/snx3000/yzhu/merge/res/merged_ /scratch/snx3000/yzhu/render/res/rendered_