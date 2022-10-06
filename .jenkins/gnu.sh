#!/bin/bash -l

### user = jenkssl
#SBATCH --constraint=gpu
#SBATCH --partition=cscsci
#SBATCH --nodes=1
###SBATCH --ntasks-per-node=1
###SBATCH --cpus-per-task=1
###SBATCH --export=ALL

set -o errexit
set -o nounset
set -o pipefail

#{{{ pe
module swap PrgEnv-cray PrgEnv-gnu
module load cdt/22.05
module load nvhpc-nompi/22.2
module load cray-hdf5-parallel/1.12.1.3
module load reframe-cscs-tests
module load daint-gpu
module load CMake
module list -t
# module unload cray-libsci_acc
# export PATH=/project/c32/src/cmake-3.24.2-linux-x86_64/bin:$PATH
# CMAKE=/apps/daint/UES/jenkins/7.0.UP03/21.09/daint-gpu/software/CMake/3.22.1/bin/cmake
# CMAKE="echo #"
CMAKE=cmake
CC --version ;echo
nvcc --version ; echo
$CMAKE --version ;echo
#}}}

#{{{ rundir
set -o xtrace   # do not set earlier to avoid noise from module
umask 0002      # make sure group members can access the data
RUNDIR=$SCRATCH/$BUILD_TAG.gnu
echo "# WORKSPACE=$WORKSPACE"
echo "# RUNDIR=$RUNDIR"
mkdir -p "$RUNDIR"
chmod 0775 "$RUNDIR"
cd "$RUNDIR"
INSTALLDIR=$RUNDIR/local
# pwd
# ls -la
#}}}

#{{{ build
# sed -i "s@GIT_REPOSITORY@SOURCE_DIR $VV/\n#@" ./cmake/setup_GTest.cmake
# sed -i "s@GIT_REPOSITORY@SOURCE_DIR $VV/\n#@" ./domain/cmake/setup_GTest.cmake
# sed -i "s@GIT_REPOSITORY@SOURCE_DIR $VV/\n#@" ./ryoanji/cmake/setup_GTest.cmake
# sed -i "s@GIT_TAG@#GIT_TAG @"                 ./cmake/setup_GTest.cmake
# sed -i "s@GIT_TAG@#GIT_TAG @"                 ./domain/cmake/setup_GTest.cmake
# sed -i "s@GIT_TAG@#GIT_TAG @"                 ./ryoanji/cmake/setup_GTest.cmake

# can be run manually with: 
# BUILD_TAG=jenkins WORKSPACE=SPH-EXA.git STAGE_NAME=log ./gnu.sh
rm -fr build
$CMAKE \
-S "${WORKSPACE}" \
-B build \
-DCMAKE_CXX_COMPILER=CC \
-DCMAKE_C_COMPILER=cc \
-DBUILD_TESTING=ON \
-DBUILD_ANALYTICAL=ON \
-DGPU_DIRECT=OFF \
-DCMAKE_CUDA_FLAGS='-arch=sm_60' \
-DCMAKE_BUILD_TYPE=Debug \
-DCMAKE_INSTALL_PREFIX=$INSTALLDIR

# TODO: -DGPU_DIRECT=ON \
#       -DSPH_EXA_WITH_H5PART=ON \

$CMAKE --build build -j 12 |& tee -a "${STAGE_NAME}.out"
# $CMAKE --build build -t sphexa -j 12 |& tee -a "${STAGE_NAME}.out"

$CMAKE --install build |& tee -a "${STAGE_NAME}.out"

# find $PWD/local -type f |& tee -a "${STAGE_NAME}.out"
#}}}

#{{{ run
# tar xf /scratch/snx3000/piccinal/jenkins.gnu/local.tar # local/bin/

#{{{ 1 compute node jobs:
RFM_TRAP_JOB_ERRORS=1 reframe -r \
--keep-stage-files \
-c $WORKSPACE/.jenkins/reframe_ci.py \
--system daint:gpu \
-n ci_unittests \
-n ci_cputests \
-n ci_gputests \
-J p=cscsci \
-S image=$INSTALLDIR
#}}}

#{{{ 2 compute node jobs: cscsci partition limited to 1cn jobs -> use -pdebug
RFM_TRAP_JOB_ERRORS=1 reframe -r \
--keep-stage-files \
-c $WORKSPACE/.jenkins/reframe_ci.py \
--system daint:gpu \
-n ci_2cn \
-J p=debug \
-S image=$INSTALLDIR
#}}}

#}}}
