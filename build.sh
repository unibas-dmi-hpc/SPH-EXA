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
module load CrayGNU/.20.08
module load craype-accel-nvidia60
module load nvhpc
export CRAYPE_LINK_TYPE=dynamic
export OMP_NUM_THREADS=1
module rm xalt
module list -t
make -f Makefile -j 2 CC="cc" CXX="CC" FC="ftn" NVCC="nvcc" mpi+omp+cuda MPICXX=CC SRCDIR=. BUILDDIR=build BINDIR=bin CUDA_PATH=$CUDATOOLKIT_HOME
mv bin/mpi+omp+cuda.app bin/mpi+omp+cuda
