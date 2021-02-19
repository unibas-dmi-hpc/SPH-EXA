#!/bin/bash

export OMP_NUM_THREADS=4

make clean
make -f Makefile -j 2 mpi+omp SRCDIR=. BUILDDIR=build BINDIR=bin NVCC="nvcc" NVCCARCH=sm_52 TESTCASE=evrard-sfc
mv bin/* bin/app
