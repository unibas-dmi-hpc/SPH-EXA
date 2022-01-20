![License](https://img.shields.io/github/license/unibas-dmi-hpc/SPH-EXA_mini-app)
[![Unit tests](https://github.com/unibas-dmi-hpc/SPH-EXA_mini-app/actions/workflows/unittest.yml/badge.svg?branch=develop)](https://github.com/unibas-dmi-hpc/SPH-EXA_mini-app/actions/workflows/unittest.yml)
![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/unibas-dmi-hpc/SPH-EXA_mini-app?include_prereleases)
<p align="center">
  <img src="https://github.com/unibas-dmi-hpc/SPH-EXA/blob/master/docs/artwork/SPH-EXA_logo.png?raw=true" alt="SPH-EXA logo"/>
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

SPH-EXA is a C++17 headers-only code with no external software dependencies.
The parallelism is currently expressed via the following models: MPI, OpenMP, CUDA and HIP.

[Check our wiki for more details](https://github.com/unibas-dmi-hpc/SPH-EXA_mini-app/wiki)

#### Folder structure

```
SPH-EXA
├── README.md
├── docs
├── domain                           - cornerstone octree and domain
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
│   └── test                        - cornerstone unit- performance-
│       ├── integration_mpi           and integration tests
│       ├── performance
│       ├── unit
│       └── unit_cuda
├── ryoanji                         - GPU N-body solver for gravity
│   ├─── src
│   └─── test                       - demonstrator app and unit tests
├── include
│   └─── sph                        - SPH kernel functions
│        ├── cuda
│        └─── kernel
├── scripts
├── src                             - test case main functions
│   ├── evrard
│   ├── sedov
│   └── sqpatch
├── test
└── tools
```
#### Compile

Use the following commands to compile the Sedov blast wave example:

Minimal CMake configuration:
```shell
mkdir build
cd build
cmake <GIT_SOURCE_DIR>
```

Recommended CMake configuration on Piz Daint:
```shell
module load daint-gpu
module load cudatoolkit
module load CMake/3.21.3 # or newer

mkdir build
cd build
cmake -DCMAKE_CXX_COMPILER=CC <GIT_SOURCE_DIR>
```

* Build everything: ```make -j```
* MPI + OpenMP: ```make sedov```
* MPI + OpenMP + CUDA: ```make sedov-cuda```


#### Running the main application

The Sedov test case binaries are located in  ```build/src/sedov/```

Possible arguments:  
* ```-n NUM``` : Run the simulation with NUM^3 (NUM to the cube) number of particles  
* ```-s NUM``` : Run the simulation with NUM of iterations (time-steps)  
* ```-w NUM``` : Dump particle data every NUM iterations (time-steps)  
* ```--quiet``` : Don't print any output to stdout  

Example usage:  
* ```OMP_NUM_THREADS=4 ./src/sedov/sedov -n 100 -s 1000 -w 10```
  Runs Sedov with 1 million particles for 1000 iterations (time-steps) with 4 OpenMP
  threads and dumps particles data every 10 iterations
* ```OMP_NUM_THREADS=4 ./src/sedov/sedov-cuda -n 100 -s 1000 -w 10```
  Runs Sedov with 1 million particles for 1000 iterations (time-steps) with 4 OpenMP
  threads. Uses the GPU for most of the compute work.
* ```OMP_NUM_THREADS=4 mpiexec -np 2 ./src/sedov/sedov -n 100 -s 1000 -w 10```
  Runs Sedov with 1 million particles for 1000 iterations (time-steps) with 2 MPI ranks of 4 OpenMP
  threads each. Works when using MPICH. For OpenMPI, use ```mpirun```  instead.
* ```OMP_NUM_THREADS=12 srun -Cgpu -A<your account> -n<nnodes> -c12 ./src/sedov/sedov-cuda -n 100 -s 1000 -w 10```
  Optimal runtime configuration on Piz Daint for `nnodes` GPU compute nodes. Launches 1 MPI rank with
  12 OpenMP threads per node.

#### Running the tests

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

## Ryoanji GPU N-body solver

Ryoanji is a high-performance GPU N-body solver for gravity. It relies on the cornerstone octree framework
for tree construction, [EXAFMM](https://github.com/exafmm/exafmm) multipole kernels,
and a warp-aware tree-traversal inspired by the
[Bonsai](https://github.com/treecode/Bonsai) GPU tree-code.

#### Compilation

```shell
cmake -DCMAKE_CXX_COMPILER=CC -DBUILD_RYOANJI=ON <GIT_SOURCE_DIR>
cd ryoanji
make -j
```

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
   * [SPH-EXA project 1](https://www.pasc-ch.org/projects/2017-2020/sph-exa/)
   * [SPH-EXA project 2](https://www.pasc-ch.org/projects/2021-2024/sph-exa2/)
* [Swiss National Supercomputing Center (CSCS)](https://www.cscs.ch/)
* [Scientific Computing Center of the University of Basel (sciCORE)](https://scicore.unibas.ch/)
