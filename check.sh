#!/bin/bash

export CRAYPE_LINK_TYPE=dynamic
export OMP_NUM_THREADS=4

make -f Makefile clean

make -f Makefile -j 2 mpi+omp SRCDIR=. BUILDDIR=build BINDIR=bin NVCC="nvcc" NVCCARCH=sm_52 TESTCASE=sedov
mv bin/* bin/app

mpirun -n 2 ./bin/app --input bigfiles/Test3DEvrardRel.bin -s 100 -w 100
mv dump_Sedov100.bin output_1.bin
rm -f dump_Sedov*

make -f Makefile clean
make -f Makefile -j 2 mpi+omp SRCDIR=. BUILDDIR=build BINDIR=bin NVCC="nvcc" NVCCARCH=sm_52 TESTCASE=sedov-sfc
mv bin/* bin/app

mpirun -n 2 ./bin/app --input bigfiles/Test3DEvrardRel.bin -s 100 -w 100
mv dump_Sedov100.bin output_2.bin
rm -f dump_Sedov*

python3 diff_output.py output_1.bin output_2.bin
