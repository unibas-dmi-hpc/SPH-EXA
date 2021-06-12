

# Cornerstone octree - a distributed domain and octree for N-body simulations

Cornerstone octree is a collection of header only routines for
* **3D Morton codes:** encoding/decoding in 32 and 64 bits
* **Octrees:** local and distributed octree builds from x,y,z coordinates, using
  a lean format that stores only the leaf nodes with one Morton code per leaf
  in a contiguous array.
* **Halo discovery:** identify halo nodes in a global octree, using a 3D collision detection
  algorithm that builds and traverses a binary radix tree.
* **Neighbor searching:** Find particle neighbors within a radius using sorted particle Morton codes
* **Particle exchange:** exchange elements of local coordinate arrays to whip them into shape
   as defined by a global octree, including halo particles.

All of these components are combined into a **global domain** to manage distributed particles
unified under a global octree. But of course, each component can also be used on its own to
build domains with different behaviors.

Cornerstone octree is a C++17 headers-only library with no external software dependencies,
except for the Google test framework, used to compile the unit tests and automatically downloaded by CMake.
Building client  applications using the libraries does not require any dependencies.

#### Folder structure

```
cornerstone-octree
|   README.md
└───include/cstone/       - folder containing all domain and octree functionality
└───test/
    └───integration_mpi/  - MPI-enabled integration tests between different units
    └───performance/      - performance test cases
    └───unit/             - (non-MPI) unit tests
    └───unit_cuda/        - CUDA unit tests
```
#### Compile

Tests have to be built with CMake, as they depend on the Google test framework
which is automatically downloaded by CMake.

Client applications can be build with any build system, with and without MPI functionality.

#### Run

A minimal sketch of what a client program employing the full domain might look like is listed below.

```c++
#include "domain.hpp"

using Real    = double;
using KeyType = unsigned;

int main()
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    
    // fill x,y,z,h with some initial values on each rank
    std::vector<Real> x{/*...*/}, y{/*...*/}, z{/*...*/}, h{/*...*/};
    
    int bucketSize = 10;
    Domain<KeyType, Real> domain(rank, numRanks, bucketSize) ;
    
    int nIterations = 10;
    for (int iter = 0; iter < nIterations; ++iter)
    {
        domain.sync(x,y,z,h);

        // x,y,z,h now contain all particles of a part of the global octree,
        // including their halos.

        std::vector<Real> density(x.size());

        // compute physical quantities, e.g. densities for particles in the assigned ranges:
        // computeDensity(density,x,y,z,h,domain.startIndex(),domain.endIndex());

        // repeat the halo exchange for densities
        domain.exchangeHalos(density);

        // compute more quantities and finally update particle positions in x,y,z and h,
        // then start over
    }

    return;
}
```

## Authors

* **Sebastian Keller**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Thanks / see also

* [PASC SPH-EXA project](https://github.com/unibas-dmi-hpc/SPH-EXA_mini-app)
* Aurélien Cavelan [@acavelan](https://github.com/acavelan) for his recursive/heap-allocation based octree
 implementation which served as a precursor for this work.
* This [blog post](https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/) about decoding Morton codes
* This NVidia developer [blog post trilogy](https://developer.nvidia.com/blog/thinking-parallel-part-i-collision-detection-gpu/)
  on 3D collision detection
