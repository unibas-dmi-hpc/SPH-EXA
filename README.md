![License](https://img.shields.io/github/license/unibas-dmi-hpc/SPH-EXA_mini-app)
[![Unit tests](https://github.com/unibas-dmi-hpc/SPH-EXA_mini-app/actions/workflows/unittest.yml/badge.svg?branch=develop)](https://github.com/unibas-dmi-hpc/SPH-EXA_mini-app/actions/workflows/unittest.yml)
![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/unibas-dmi-hpc/SPH-EXA_mini-app?include_prereleases)
<p align="center">
  <img src="https://raw.githubusercontent.com/unibas-dmi-hpc/SPH-EXA/develop/docs/artwork/SPH-EXA_logo.png" alt="SPH-EXA logo" width="200"/>
</p>

# SPH

The smoothed particle hydrodynamics (SPH) technique is a purely Lagrangian method.
SPH discretizes a fluid in a series of interpolation points (SPH particles)
whose distribution follows the mass density of the fluid and their evolution relies
on a weighted interpolation over close neighboring particles.

SPH simulations represent computationally demanding calculations.
Therefore, trade-offs are made between temporal and spatial scales, resolution,
dimensionality (3-D or 2-D), and approximated versions of the physics involved.
The parallelization of SPH codes is not trivial due to their boundless nature
and the absence of a structured grid.
[SPHYNX](https://astro.physik.unibas.ch/sphynx/),
[ChaNGa](http://faculty.washington.edu/trq/hpcc/tools/changa.html),
and [SPH-flow](http://www.sph-flow.com) are the three SPH codes selected in the PASC SPH-EXA project to
act as parent and reference codes to SPH-EXA.
The performance of these three codes is negatively impacted by factors such as imbalanced multi-scale physics, individual time-stepping, halos exchange, and long-range forces.
Therefore, the goal is to extrapolate their common basic SPH features, and consolidate them in a fully optimized, Exascale-ready, MPI+X, SPH code: SPH-EXA.

# SPH-EXA

SPH-EXA is a C++17 simulation code, parallelized with MPI, OpenMP, CUDA and HIP.

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
└── src
    ├── init                        - initial conditions for test cases
    ├── io                          - file output functionality
    └── sphexa                      - SPH main application front-end
```
#### Compile

Use the following commands to compile the main SPH-EXA application:

Minimal CMake configuration:
```shell
mkdir build
cd build
cmake <GIT_SOURCE_DIR>
```

Recommended CMake configuration on Piz Daint,
using the (default) Cray Clang compiler for CPU code (.cpp) and nvcc/g++ for GPU code (.cu):
```shell
module load daint-gpu
module load cudatoolkit/11.2.0_3.39-2.1__gf93aa1c # or newer
module load CMake/3.22.1               # or newer
module load cray-hdf5-parallel
module load gcc/9.3.0                  # nvcc uses gcc as the default host compiler,
                                       # but the system version is too old
export GCC_X86_64=/opt/gcc/9.3.0/snos  # system header versions are too old, applies to cray-clang too

mkdir build
cd build
cmake -DCMAKE_CXX_COMPILER=CC <GIT_SOURCE_DIR>
```

* Build everything: ```make -j```


#### Running the main application

The main ```sphexa``` application can either start a simulation by reading initial conditions
from a file or generate an initial configuration for a named test case.
Self-gravity will be activated automatically based on named test-case choice or if the HDF5 initial
configuration file has an HDF5 attribute with a non-zero value for the gravitational constant.

Arguments:  
* ```--init CASE/FILE ```: `sedov` for simulation the Sedov blast wave, `noh` for the Noh implosion,
                            `evrard` for the Evrard collapse or provide an HDF5 file with valid input data
* ```-n NUM``` : Run the simulation with NUM^3 (NUM to the cube) number of particles (for named test cases)
* ```-s NUM``` : Run the simulation with NUM of iterations (time-steps)  
* ```-w NUM``` : Dump particle data every NUM iterations (time-steps)  
* ```-f FIELDS```: Comma separated list of particle fields for file output dumps
* ```--quiet``` : Don't print any output to stdout  

Example usage:  
* ```OMP_NUM_THREADS=4 ./sphexa --init sedov -n 100 -s 1000 -w 10 -f x,y,z,rho,p```
  Runs Sedov with 100^3 particles for 1000 iterations (time-steps) with 4 OpenMP
  threads and dumps particle xyz-coordinates, density and pressure data every 10 iterations
* ```OMP_NUM_THREADS=4 ./sphexa-cuda --init -n 100 -s 1000 -w 10 -f x,y,z,rho,p```
  Runs Sedov with 100^3 million particles for 1000 iterations (time-steps) with 4 OpenMP
  threads. Uses the GPU for most of the compute work.
* ```OMP_NUM_THREADS=4 mpiexec -np 2 ./sphexa --init noh -n 100 -s 1000 -w 10```
  Runs Noh with 100^3 million particles for 1000 iterations (time-steps) with 2 MPI ranks of 4 OpenMP
  threads each. Works when using MPICH. For OpenMPI, use ```mpirun```  instead.
* ```OMP_NUM_THREADS=12 srun -Cgpu -A<your account> -n<nnodes> -c12 ./sphexa-cuda --init sedov -n 100 -s 1000 -w 10```
  Optimal runtime configuration on Piz Daint for `nnodes` GPU compute nodes. Launches 1 MPI rank with
  12 OpenMP threads per node.
* ```./sphexa-cuda --init evrard.h5 -s 2000 -w 100 -f x,y,z,rho,p,vx,vy,vz```
  Run SPH-EXA, initializing particle data from an input file (e.g. for the Evrard collapse). Includes
  gravitational forces between particles. The angle dependent accuracy parameter theta can be specificed
  with ```--theta <value>```, the default is `0.5`.

#### Running the unit, integration and regression tests

Cornerstone octree and domain unit tests:

```shell
./domain/test/unit/component_units
```

GPU-enabled unit tests:
```shell
./domain/test/unit_cuda/component_units_cuda
```

MPI-enabled integration and regression tests:

```shell
mpiexec -np 2 ./domain/test/integration_mpi/domain_2ranks
mpiexec -np 2 ./domain/test/integration_mpi/exchange_focus
mpiexec -np 2 ./domain/test/integration_mpi/exchange_halos
mpiexec -np 2 ./domain/test/integration_mpi/globaloctree

mpiexec -np 5 ./domain/test/integration_mpi/domain_nranks
mpiexec -np 5 ./domain/test/integration_mpi/exchange_domain
mpiexec -np 5 ./domain/test/integration_mpi/exchange_keys
mpiexec -np 5 ./domain/test/integration_mpi/focus_tree
mpiexec -np 5 ./domain/test/integration_mpi/treedomain
```

SPH-kernel unit tests:

```shell
./include/sph/test/kernel/kernel_tests
```
## Input data

Some tests require input data. 
For example, the Evrard test case will check that a Test3DEvrardRel.bin file exists and can be read at the beginning of the job.
 This file can be downloaded from [zenodo.org](https://zenodo.org/record/4904876/files/Test3DEvrardRel.dat.gz?download=1).

## Ryoanji GPU N-body solver

Ryoanji is a high-performance GPU N-body solver for gravity. It relies on the cornerstone octree framework
for tree construction, [EXAFMM](https://github.com/exafmm/exafmm) multipole kernels,
and a warp-aware tree-traversal inspired by the
[Bonsai](https://github.com/treecode/Bonsai) GPU tree-code.

#### Running the demonstrator app
```shell
./test/ryoanji <log2(numParticles)> <computeDirectReference:yes=1,no=0>
```

#### Running the unit tests
```shell
./test/ryoanji_unit_tests
```

## Authors (in alphabetical order)

* Ruben Cabezon
* Aurelien Cavelan
* Florina Ciorba
* Michal Grabarczyk
* Danilo Guerrera
* David Imbert
* Sebastian Keller
* Lucio Mayer
* Ali Mohammed
* Jg Piccinali
* Tom Quinn
* Darren Reed

## Paper references
[Cavelan, A., Cabezon, R. M., Grabarczyk, M., Ciorba, F. M. (2020). A Smoothed Particle Hydrodynamics Mini-App for Exascale. Proceedings of the Platform for Advanced Scientific Computing Conference (PASC '20). Association for Computing Machinery. DOI: 10.1145/3394277.3401855](https://dl.acm.org/doi/10.1145/3394277.3401855)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* [Platform for Advanced Scientific Computing (PASC)](https://www.pasc-ch.org/)
   * [SPH-EXA project 1](https://www.pasc-ch.org/projects/2017-2020/sph-exa/) and [webpage](https://hpc.dmi.unibas.ch/en/research/sph-exa/)
   * [SPH-EXA project 2](https://www.pasc-ch.org/projects/2021-2024/sph-exa2/) and [webpage](https://hpc.dmi.unibas.ch/en/research/pasc-sph-exa2/)
* [Swiss National Supercomputing Center (CSCS)](https://www.cscs.ch/)
* [Scientific Computing Center of the University of Basel (sciCORE)](https://scicore.unibas.ch/)
* [Swiss participation in Square Kilometer Aray (SKA)](https://www.sbfi.admin.ch/sbfi/en/home/research-and-innovation/international-cooperation-r-and-i/international-research-organisations/skao.html)
