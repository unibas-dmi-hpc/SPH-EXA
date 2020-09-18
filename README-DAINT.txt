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