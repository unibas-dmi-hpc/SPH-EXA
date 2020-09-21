# With ssh
git clone git@github.com:unibas-dmi-hpc/SPH-EXA_mini-app.git
# With HTTPS
git clone https://github.com/unibas-dmi-hpc/SPH-EXA_mini-app.git

# Cray-Clang
module load daint-gpu
module load cudatoolkit
module load craype-accel-nvidia60

make TESTCASE=sedov NVCC=nvcc ENV=clang MPICXX=CC mpi+omp
make TESTCASE=sedov NVCC=nvcc ENV=clang MPICXX=CC mpi+omp+target
make TESTCASE=sedov NVCC=nvcc ENV=clang MPICXX=CC mpi+omp+cuda

# Pgi
module unload PrgEnv-cray
module load PrgEnv-pgi
module load craype-accel-nvidia60
export CUDA_HOME=$CUDATOOLKIT_HOME

make TESTCASE=sedov NVCC=nvcc ENV=pgi MPICXX=CC mpi+omp+acc

# To allocate an interactive session:
(using c16 project)
salloc -A c16 -C gpu --nodes=1  --exclusive --ntasks-per-core=2 --cpus-per-task=24 --partition=normal --time 02:00:00 bash

(using hackathon account)
salloc -C gpu --nodes=1  --exclusive --ntasks-per-core=2 --cpus-per-task=24 --partition=debug --time 00:30:00 bash


# To run:
srun bin/mpi+omp+cuda.app -n 100 -s 200 -w 100

where n is the size of the cube (# of particles in each dimension), s is the number of timesteps, w is the frequency to write a snapshot in timesteps.

# To do profiling
srun nvprof bin/mpi+omp+cuda.app -n 100 -s 0
srun cuda-memcheck bin/mpi+omp+cuda.app -n 20 -s 0

(-s 0 will run one iteration)

# python
python plot.py dump_Sedov0.bin 1000000

# gnuplot
set size square;
pl 'dump_Sedov200.txt' u (abs($3/$7)<2.?$1:1/0):2:8 w p pt 7 lc palette z
pl 'dump_Sedov200.txt' u (($1*$1+$2*$2+$3*$3)**.5):8  w d
