# Compile SPH-EXA with ADIOS2

Typically, the default ADIOS2 installation on clusters lacks optional plugins. Therefore, to enable SPH-EXA for its compression capabilities, it is necessary to build ADIOS2 independently. The instructions below detail the process of building ADIOS2 on specific platforms, and subsequently linking SPH-EXA with it.

## On Piz Daint

Assuming a clean environment (i.e., no user-specific environmental variables are configured), load the following modules:
```bash
module load PrgEnv-cray
module load cdt/22.05 # will load cce/14.0.0
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/22.2/cuda/11.6/bin:$PATH
module load cray-hdf5-parallel
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:LD_LIBRARY_PATH
```
### Install Spack and ADIOS2

Install Spack using the provided commands:
```bash
git clone -c feature.manyFiles=true https://github.com/spack/spack.git ~/spack
```

Activate Spack command-line tools:
```bash
. ~/spack/share/spack/setup-env.sh
```

Automatically search for existing packages in the current environment:
```bash
spack external find cuda hdf5 cmake openssl zlib bzip2 pkg-config
```

Detect native compilers using Spack (refer to [Spack on Cray](https://spack.readthedocs.io/en/latest/getting_started.html#spack-on-cray) for more details):
```bash
spack compiler find
```

Assuming we use cce/14.0.0 as main compiler, open the Spack package config file:
```bash
vi ~/.spack/packages.yaml
```

Insert the following lines *at the beginning of* packages.yaml, immediately after `packages:`:
```yaml
  cray-mpich:
    externals:
    - spec: "cray-mpich@7.7.20%cce@14.0.0"
      modules:
      - cray-mpich/7.7.20
    buildable: false
  all:
    compiler: [cce@14.0.0]
    providers:
      mpi: [cray-mpich]
```

Proceed to install ADIOS2 with the following specifications. Note that there is currently no valid compilation with MGARD on Spack due to it not adhering to the C++17 standard.

```bash
spack install --reuse adios2@2.9.2~aws+blosc2+bzip2+cuda~dataman~dataspaces~fortran+hdf5~ipo~kokkos+libcatalyst+libpressio~mgard+mpi+pic+png+python~rocm+shared+sst+sz+zfp ^hdf5@1.12.1 ^openblas~fortran
```

Given the provided configuration, the installation process is estimated to take ~45 minutes.
For the convenience of subsequent SPH-EXA building and execution, consider appending the following lines to the end of your `~/.bashrc` file:

```bash
. ~/spack/share/spack/setup-env.sh
spack load adios2
```
### Integration with SPH-EXA

Ensure that you have the following environmental setup activated before proceeding:
```bash
module load PrgEnv-cray
module load cdt/22.05 # will load cce/14.0.0
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/22.2/cuda/11.6/bin:$PATH
module load cray-hdf5-parallel
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:LD_LIBRARY_PATH
. ~/spack/share/spack/setup-env.sh
spack load adios2
```

Then download the code. Run CMake with following flags:
```bash
CC=cc CXX=CC cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_CUDA_ARCHITECTURES=60 -DSPH_EXA_WITH_H5PART=ON -DSPH_EXA_WITH_ADIOS=ON -DADIOS_WITH_MGARD=OFF -S ../SPH-EXA
```