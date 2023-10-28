#!/bin/bash -l
#
#
# 8 nodes, 8 MPI task per node
#
#SBATCH --job-name="render"
#SBATCH --time=4:00:00
#SBATCH --nodes=10
#SBATCH --ntasks=20
#SBATCH --constraint=gpu
#SBATCH --account=c32
#========================================


# To run this you need a python environment with:
# - Pillow
# - matplotlib
# - numpy
# No need for parallellization. It's fast enough (on a proper vis GPU)
module load daint-gpu
module load h5py numpy matplotlib

# source /scratch/snx3000/yzhu/render/venv/bin/activate

srun --cpu_bind=sockets python render.py parallel /scratch/snx3000/yzhu/merge/res/merged_3000_ /scratch/snx3000/yzhu/render/res/rendered_