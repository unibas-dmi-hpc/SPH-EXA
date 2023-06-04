#!/usr/bin/env sh


# Replace $1 with the parent directory of SPH-EXA
SPHEXA_ROOT="$1/../"
cd $SPHEXA_ROOT/sphexa-buildcatalyst
. ~/spack/share/spack/setup-env.sh
spack load conduit python py-numpy py-mpi4py paraview
spack unload mpi
cd $SPHEXA_ROOT/sphexa-buildcatalyst
# rm -rf ./*
cmake -DParaView_CATALYST_DIR=$(spack location --install-dir paraview)/lib/catalyst/ -DCMAKE_BUILD_TYPE=Debug -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_CXX_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DINSITU=Catalyst -DSPH_EXA_WITH_H5PART:BOOL=OFF -S ../SPH-EXA
cd $SPHEXA_ROOT/sphexa-buildcatalyst
make -j 16

# Images will be generated in $SPHEXA_ROOT/sphexa-vis/output_catalyst. You can clear it before running.
rm -f $SPHEXA_ROOT/sphexa-vis/output_catalyst/*

# Run the testcase
# mpiexec -n 1 $1/sphexa-build/main/src/sphexa/sphexa --init sedov -s 101 -n 30 --prop std --ascii --quiet --catalyst $1/SPH-EXA/scripts/catalyst/helloworld.py