#!/usr/bin/env sh

# Replace $1 with the parent directory of SPH-EXA
SPHEXA_ROOT="$1"
. ~/spack/share/spack/setup-env.sh
spack load ascent cmake
cd $SPHEXA_ROOT/sphexa-vis/ascent-build
# rm -rf ./*
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_CXX_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_CUDA_ARCHITECTURES=60 -DINSITU=Ascent -DAscent_DIR=$(spack location --install-dir ascent)/lib/cmake/ascent -S ../../SPH-EXA
cd $SPHEXA_ROOT/sphexa-vis/ascent-build
make -j 16
# Images will be generated in $SPHEXA_ROOT/sphexa-vis/output/ascent/. You can clear it before running.
rm -f $SPHEXA_ROOT/sphexa-vis/output/ascent/*

# Run the testcase
# mpiexec -n 4 $SPHEXA_ROOT/sphexa-build/main/src/sphexa/sphexa --ascii --init sedov -s 1000 -n 30 --prop std --quiet

# mpiexec -n 4 /home/appcell/unibas/sphexa-build/main/src/sphexa/sphexa --ascii --init sedov -s 1001 -n 30 --prop std --quiet