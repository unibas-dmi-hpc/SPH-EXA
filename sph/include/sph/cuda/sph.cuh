#pragma once

#include "cstone/sfc/box.hpp"
#include "cstone/cuda/findneighbors.cuh"

#include "sph/particles_data.hpp"

#include "gpu_particle_data.cuh"
#include "cuda_utils.cuh"

//! @brief maximum number of neighbors supported in GPU kernels
#define NGMAX 150

namespace sphexa
{
namespace sph
{
namespace cuda
{

struct CudaConfig
{
    //! @brief number of threads per block for the traversal kernel
    static constexpr int numThreads = 128;
};

template<class Dataset>
extern void computeRhoZero(
    size_t, size_t, size_t, Dataset& d, const cstone::Box<typename Dataset::RealType>&);

template<class Dataset>
extern void computeDensity(
    size_t, size_t, size_t, Dataset& d, const cstone::Box<typename Dataset::RealType>&);

template<class Dataset>
extern void computeIAD(
    size_t, size_t, size_t, Dataset& d, const cstone::Box<typename Dataset::RealType>&);

template<class Dataset>
extern void computeDivvCurlv(
    size_t, size_t, size_t, Dataset& d, const cstone::Box<typename Dataset::RealType>&);

template<class Dataset>
extern void computeAVswitches(
    size_t, size_t, size_t, Dataset& d, const cstone::Box<typename Dataset::RealType>&);

template<class Dataset>
extern void computeMomentumEnergy(
    size_t, size_t, size_t, Dataset& d, const cstone::Box<typename Dataset::RealType>&);

} // namespace cuda
} // namespace sph
} // namespace sphexa
