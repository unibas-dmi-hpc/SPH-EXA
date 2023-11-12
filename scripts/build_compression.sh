#!/usr/bin/env sh

# Replace $1 with the parent directory of SPH-EXA
SPHEXA_ROOT="$1"
cd $SPHEXA_ROOT/sphexa-vis/compression-build
export HDF5_PLUGIN_PATH=/home/appcell/unibas/zfpbuild/plugin
spack load hdf5/zwotjod
# rm -rf ./*
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_CXX_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_CUDA_ARCHITECTURES=60 -DSPH_EXA_WITH_H5PART=ON -DSPH_EXA_WITH_HDF5=ON -S $SPHEXA_ROOT/SPH-EXA
make -j16 sphexa

# Images will be generated in $SPHEXA_ROOT/sphexa-vis/output. You can clear it before running.
# rm -f $SPHEXA_ROOT/sphexa-vis/output/compression/*