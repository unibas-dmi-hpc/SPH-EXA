#!/usr/bin/env sh


# Building with local ParaView
SPHEXA_ROOT="$1"
. ~/spack/share/spack/setup-env.sh
spack load ascent
cd $SPHEXA_ROOT/sphexa-vis/catalyst-build
rm -rf ./*
cmake -Dcatalyst_DIR=$SPHEXA_ROOT/paraview/install/lib/cmake/catalyst-2.0 -DParaView_CATALYST_DIR=$SPHEXA_ROOT/paraview/install/lib/catalyst -DCMAKE_BUILD_TYPE=Debug -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_CXX_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DINSITU=Catalyst -DSPH_EXA_WITH_H5PART:BOOL=ON -S ../../SPH-EXA
cd $SPHEXA_ROOT/sphexa-vis/catalyst-build
make -j 16
rm -f $SPHEXA_ROOT/sphexa-vis/output/catalyst/*


# Run the testcase
# mpiexec -n 1 $1/sphexa-build/main/src/sphexa/sphexa --init sedov -s 101 -n 30 --prop std --ascii --quiet --catalyst $1/SPH-EXA/scripts/catalyst/threeplanes.py