#!/usr/bin/env sh

# Replace $1 with the parent directory of SPH-EXA
SPHEXA_ROOT="$1"
cd $SPHEXA_ROOT/sphexa-zfp
export HDF5_PLUGIN_PATH=/home/appcell/unibas/zfpbuild/plugin
# rm -rf ./*
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_CXX_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_CUDA_ARCHITECTURES=60 -DSPH_EXA_WITH_H5PART=ON -DSPH_EXA_WITH_HDF5=ON -S ../SPH-EXA
cd $SPHEXA_ROOT/sphexa-zfp
make -j16 sphexa

# Images will be generated in $SPHEXA_ROOT/sphexa-vis/output. You can clear it before running.
rm -f $SPHEXA_ROOT/sphexa-vis/output_zfp/*