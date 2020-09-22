#!/bin/bash

_onerror()
{
    exitcode=$?
    echo "-reframe: command \`$BASH_COMMAND' failed (exit code: $exitcode)"
    exit $exitcode
}

trap _onerror ERR

module load daint-gpu
module unload PrgEnv-cray
module load PrgEnv-gnu
module load craype-accel-nvidia60
export CRAYPE_LINK_TYPE=dynamic
export OMP_NUM_THREADS=1
module rm xalt
make -f Makefile -j 2 CC="cc" CXX="CC" FC="ftn" NVCC="nvcc -g -G" CXXFLAGS="-std=c++14 -g -O0 -DNDEBUG" mpi+omp+cuda SRCDIR=. BUILDDIR=build BINDIR=bin NVCCFLAGS="-std=c++14 -rdc=true -arch=sm_60 -g -G --expt-relaxed-constexpr" NVCCARCH=sm_60 CUDA_PATH=$CUDA_PATH MPICXX=CC
mv bin/mpi+omp+cuda.app bin/mpi+omp+cuda
