#!/usr/bin/env sh

# Replace $1 with the parent directory of SPH-EXA
. ~/spack/share/spack/setup-env.sh
spack load conduit python py-numpy py-mpi4py paraview
spack unload mpi
cd $1/sphexa-build
rm -rf ./*
cmake -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_CXX_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DINSITU=Ascent -DAscent_DIR=$(spack location --install-dir ascent)/lib/cmake/ascent -S ../SPH-EXA
cd $1/sphexa-build
make -j 16

# Images will be generated in $1/sphexa-vis. You can clear it before running.
# cd $1/sphexa-vis
# rm -f $1/sphexa-vis/output/*

# Run the testcase
# mpiexec -n 1 $1/sphexa-build/main/src/sphexa/sphexa --ascii --init sedov -s 5 -n 30 --prop std