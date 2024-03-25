![License](https://img.shields.io/github/license/unibas-dmi-hpc/SPH-EXA_mini-app)
[![Documentation Status](https://readthedocs.org/projects/sph-exa/badge/?version=latest)](https://sph-exa.readthedocs.io/en/latest/?badge=latest)
[![Unit tests](https://github.com/unibas-dmi-hpc/SPH-EXA_mini-app/actions/workflows/unittest.yml/badge.svg?branch=develop)](https://github.com/unibas-dmi-hpc/SPH-EXA_mini-app/actions/workflows/unittest.yml)
![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/unibas-dmi-hpc/SPH-EXA_mini-app?include_prereleases)
<p align="center">
  <img src="https://raw.githubusercontent.com/unibas-dmi-hpc/SPH-EXA/develop/docs/artwork/SPH-EXA_logo.png" alt="SPH-EXA logo" width="200"/>
</p>

# SPH

The smoothed particle hydrodynamics (SPH) technique is a purely Lagrangian method.
SPH discretizes a fluid in a series of interpolation points (SPH particles) whose distribution follows the mass density of the fluid and their evolution relies
on a weighted interpolation over close neighboring particles.

The parallelization of SPH codes is not trivial due to their boundless nature and the absence of a structured grid.

# SPH-EXA

SPH-EXA is a C++20 simulation code for hydrodynamics simulations (with gravity and other physics), parallelized with MPI, OpenMP, CUDA, and HIP.

SPH-EXA is built with high performance, scalability, portability, and resilience in mind. Its SPH implementation is based on [SPHYNX](https://astro.physik.unibas.ch/sphynx/), [ChaNGa](http://faculty.washington.edu/trq/hpcc/tools/changa.html), and [SPH-flow](http://www.sph-flow.com), three SPH codes selected in the PASC SPH-EXA project to act as parent and reference codes to SPH-EXA.

The performance of standard codes is negatively impacted by factors such as imbalanced multi-scale physics, individual time-stepping, halos exchange, and long-range forces. Therefore, the goal is to extrapolate common basic SPH features, and consolidate them in a fully optimized, Exascale-ready, MPI+X, SPH code: SPH-EXA.

[Check our wiki for more details](https://github.com/unibas-dmi-hpc/SPH-EXA_mini-app/wiki)

#### Folder structure

```
SPH-EXA
├── README.md
├── docs
├── domain                           - Cornerstone library: octree building and domain decomposition
│   ├── include
│   │   └── cstone
│   │       ├── CMakeLists.txt
│   │       ├── cuda
│   │       ├── domain
│   │       ├── findneighbors.hpp
│   │       ├── halos
│   │       ├── primitives
│   │       ├── sfc
│   │       ├── tree
│   │       └── util
│   └── test                        - Cornerstone unit- performance-
│       ├── integration_mpi           and integration tests
│       ├── performance
│       ├── unit
│       └── unit_cuda
├── ryoanji                         - Ryoanji: N-body solver for gravity
│   ├─── src
│   └─── test                       - demonstrator app and unit tests
│
├── sph                             - SPH implementation
│   ├─── include
│   │    └── sph
│   └─── test                       - SPH kernel unit tests
│
└── main/src
    ├── init                        - initial conditions for test cases
    ├── io                          - file output functionality
    └── sphexa                      - SPH main application front-end
```
#### Toolchain requirements

The C++ (.cpp) part of the code requires a **C++20 compiler**, at least GCC 11, clang 12 or cray-clang 14.
For CUDA (.cu), the minimum supported CUDA version is **CUDA 11.2** with a C++17 host compiler, e.g. GCC 9.3.0.

For ease of use, the recommended minimum version of CUDA is 11.4.1 which supports GCC 11, providing both the required
C++20 support and bug-free CUDA host compilation. [**NOTE:** CUDA/11.3.1 seems to have solved the compatibility issues with GCC 10.3.0]

#### Compilation

Minimal CMake configuration:
```shell
mkdir build
cd build
cmake <GIT_SOURCE_DIR>
```
Compilation at sciCORE (UniBas):
```shell
ml HDF5/1.14.2-gompi-2022a-zen2
ml CMake/3.23.1-GCCcore-11.3.0
ml CUDA/11.8.0

mkdir build
cd build
cmake <GIT_SOURCE_DIR>
```
CMake configuration on Piz Daint for clang:
**Cray-clang 14** for CPU code (.cpp), **CUDA 11.6 + GCC 11.2.0** for GPU code (.cu):
```shell
module load daint-gpu
module load CMake/3.22.1
module load PrgEnv-cray
module load cdt/22.05           # will load cce/14.0.0
module load nvhpc-nompi/22.2    # will load nvcc/11.6
module load gcc/11.2.0
module load cray-hdf5-parallel

mkdir build
cd build

# C-compiler is needed for hdf5 detection
CC=cc CXX=CC cmake -DCMAKE_CUDA_ARCHITECTURES=60 -S <GIT_SOURCE_DIR>

```
Module and CMake configuration on LUMI
```shell
module load CrayEnv buildtools/22.08 craype-accel-amd-gfx90a rocm cray-hdf5-parallel
cd <GIT_SOURCE_DIR>; hipify-perl -inplace `find -name *.cu -o -name *.cuh` && find -name *.prehip -delete
cmake -DCMAKE_CXX_COMPILER=CC -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_HIP_COMPILER=CC -DCMAKE_HIP_COMPILER_FORCED=ON -DGPU_DIRECT=<ON/OFF> -S <GIT_SOURCE_DIR>
```

Build everything: ```make -j```


#### Running the main application

The main ```sphexa``` (and `sphexa-cuda`, if GPUs are available) application can either start a simulation by reading initial conditions
from a file or generate an initial configuration for a named test case.
Self-gravity will be activated automatically based on named test-case choice or if the HDF5 initial
configuration file has an HDF5 attribute with a non-zero value for the gravitational constant.

Arguments:  
* ```--init CASE/FILE```: use the case name as seen below or provide an HDF5 file with initial conditions
* `--glass FILE`: template glass block for IC generation avaiable from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8369645.svg)](https://doi.org/10.5281/zenodo.8369645)
* ```-n NUM``` : Run the simulation with NUM^3 (NUM to the cube) number of particles (for named test cases). [NOTE: This might vary with the test case]
* ```-s NUM``` : Run the simulation with NUM of iterations (time-steps) if NUM is integer. Run until the specified physical time if NUM is real. 
* ```-w NUM``` : Dump particle data every NUM iterations (time-steps) if NUM is integer. Dump data at the specified physical time if NUM is real.
* ```-f FIELDS```: Comma separated list of particle fields for file output dumps. See a list of common output fields below.
* ```-compression COMPRESSOR_SETTINGS```: Comma separated list of compressor settings. Only available when [compiled with ADIOS2](README_adios.md) and using BP output. See a list of common compressor and settings below.
* ```--quiet``` : Don't print any output to stdout

Implemented cases:
* ```--sedov```: spherical blast wave
* ```--noh```: spherical implosion
* ```--evrard```: gravitational collapse of an isothermal cloud
* ```--turbulence```: subsonic turbulence in a box
* ```--kelvin-helmholtz```: development of the subsonic Kelvin-Helmholtz instability in a thin slice

Only the Sedov test case supports running without providing a glass block (`--glass`), but for accurate simulation
results, a glass block is nevertheless strongly recommended.

Common output fields:
* ```x, y, z```: position
* ```vx, vy, vz```: velocity
* ```h```: smoothing length
* ```rho```: density
* ```c```: speed of sound
* ```p```: pressure
* ```temp```: temperature
* ```u```: internal energy
* ```nc```: number of neighbors
* ```divv```: Module of the divergence of the velocity field
* ```curlv```: Module of the curl of the velocity field

Common compressor settings:

Please make sure that SPH-EXA is [compiled with ADIOS2](README_adios.md) before proceeding with compressors.

The input compressor settings should be a string separated by comma (,). Each token is further split into key and value using the equals sign (=) as the delimiter.

Compressor example usage: 
* ```--compression name=sz,accuracy=0.01```: use SZ lossy compressor for output, accuracy is 0.01

Available compressors:
* ```sz```: SZ, need SZ installed. Available fields:
  - `accuracy`: a float number, fixed absolute error tolerance
* ```cusz```: cuSZ, CUDA-based SZ. Need SZ installed and ADIOS compiled with CUDA
* ```cuszp```: cuSZp, CUDA-based accelerated SZ. Need SZ and CUDA installed, and ADIOS compiled with LibPressio
* ```zfp```: ZFP, need ZFP installed. Available fields:
  - `accuracy`: a float number, fixed absolute error tolerance.
  - `rate`: an integer, fixed number of bits in a compression unit.
  - `precision`: an integer, fixed number of uncompressed bits per value.
  **The 3 fields above are mutually exclusive.**
  - `backend`: a string, specifying backend device. Can be `cuda`, `omp` or `serial`. `omp` by default.
* ```cuzfp```: cuZFP, CUDA-based ZFP. Need SZ installed and ADIOS compiled with CUDA
* ```bzip2```: BZip2, need BZip2 installed
* ```mgard```: MGARD, need MGARD installed. See [here](https://github.com/CODARcode/MGARD/blob/master/doc/MGARD-X.md) for detailed info of params. Available fields:
  - `accuracy`: a float number, fixed absolute error tolerance.
  - `tolerance`: an integer, fixed number of bits in a compression unit.
  **The 2 fields above are mutually exclusive.**
  - `mode`: 
  - `s`: 

Example usage:  
* ```OMP_NUM_THREADS=4 ./sphexa --init sedov -n 100 -s 1000 -w 10 -f x,y,z,rho,p```
  Runs Sedov with 100^3 particles for 1000 iterations (time-steps) with 4 OpenMP
  threads and dumps particle xyz-coordinates, density and pressure data every 10 iterations
* ```OMP_NUM_THREADS=4 ./sphexa-cuda --init sedov -n 100 -s 1000 -w 10 -f x,y,z,rho,p```
  Runs Sedov with 100^3 particles for 1000 iterations (time-steps) with 4 OpenMP
  threads. Uses the GPU for most of the compute work.
* ```OMP_NUM_THREADS=4 mpiexec -np 2 ./sphexa --init noh -n 100 -s 1000 -w 10```
  Runs Noh with 100^3 particles for 1000 iterations (time-steps) with 2 MPI ranks of 4 OpenMP
  threads each. Works when using MPICH. For OpenMPI, use ```mpirun```  instead.
* ```OMP_NUM_THREADS=12 srun -Cgpu -A<your account> -n<nnodes> -c12 --hint=nomultithread ./sphexa-cuda --init sedov -n 100 -s 1000 -w 10```
  Optimal runtime configuration on Piz Daint for `nnodes` GPU compute nodes. Launches 1 MPI rank with
  12 OpenMP threads per node.
* ```./sphexa-cuda --init evrard --glass 50c.h5 -s 2000 -w 100 -f x,y,z,rho,p,vx,vy,vz```
  Run SPH-EXA, initializing particle data from an input file (e.g. for the Evrard collapse). Includes
  gravitational forces between particles. The angle dependent accuracy parameter theta can be specificed
  with ```--theta <value>```, the default is `0.5`.

#### Restarting from checkpoint files

If output to file is enabled and if the ```-f``` option is not provided, sphexa will output all conserved particle
fields which allows restoring the simulation to the exact state at the time of writing the output.
This includes the following fields ```x_m1, y_m1, z_m1, du_m1```.
In order to save diskspace, sphexa can be instructed to omit these fields by setting the ```-f option```, e.g.
```-f x,y,z,m,h,temp,alpha,vx,vy,vz```. If one wants to restart the simulation from an output file containing
these fields, it is necessary to add the ```_m1```. We provide an example script that can be used to achieve this:
```bash
./scripts/add_m1.py <hdf5-output-file>
```

#### Unit, integration and regression tests

Cornerstone octree comes with an extensive suite of unit, integration and regression tests, see [README](domain/README.md).

SPH kernel unit tests:

```shell
./sph/test/hydro_ve
./sph/test/hydro_std
```

## Input data

Some tests require template blocks with glass-like (Voronoi tesselated) particle distributions, these can be obtained here:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8369645.svg)](https://doi.org/10.5281/zenodo.8369645)

## Ryoanji GPU N-body solver

Ryoanji is a high-performance GPU N-body solver for gravity. It relies on the cornerstone octree framework
for tree construction, [EXAFMM](https://github.com/exafmm/exafmm) multipole kernels,
and a warp-aware tree-traversal inspired by the
[Bonsai](https://github.com/treecode/Bonsai) GPU tree-code.

## Authors (in alphabetical order)

* Ruben Cabezon (PI)
* Aurelien Cavelan
* Florina Ciorba (PI)
* Jonathan Coles
* Jose Escartin
* Jean M. Favre
* Sebastian Keller (lead dev)
* Noah Kubli
* Lucio Mayer (PI)
* Jg Piccinali
* Tom Quinn
* Darren Reed
* Lukas Schmid
* Osman Seckin Simsek
* Yiqing Zhu

## Paper references
* [Keller, S., Cavelan, A., Cabezon, R. M., Mayer L., Ciorba, F. M. (2023) Cornerstone: Octree Construction Algorithms for Scalable Particle Simulations. (PASC '23)](https://dl.acm.org/doi/abs/10.1145/3592979.3593417)
* [Cavelan, A., Cabezon, R. M., Grabarczyk, M., Ciorba, F. M. (2020). A Smoothed Particle Hydrodynamics Mini-App for Exascale. (PASC '20)](https://dl.acm.org/doi/10.1145/3394277.3401855)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* [Platform for Advanced Scientific Computing (PASC)](https://www.pasc-ch.org/)
   * [SPH-EXA project 1](https://www.pasc-ch.org/projects/2017-2020/sph-exa/) and [webpage](https://hpc.dmi.unibas.ch/en/research/sph-exa/)
   * [SPH-EXA project 2](https://www.pasc-ch.org/projects/2021-2024/sph-exa2/) and [webpage](https://hpc.dmi.unibas.ch/en/research/pasc-sph-exa2/)
* [Swiss National Supercomputing Center (CSCS)](https://www.cscs.ch/)
* [Scientific Computing Center of the University of Basel (sciCORE)](https://scicore.unibas.ch/)
* [Swiss participation in Square Kilometer Array (SKA)](https://www.sbfi.admin.ch/sbfi/en/home/research-and-innovation/international-cooperation-r-and-i/international-research-organisations/skao.html)
