#pragma once

#include "gpu_particle_data.cuh"
#include "cstone/sfc/box.hpp"

//! @brief maximum number of neighbors supported in GPU kernels
#define NGMAX 150

namespace sphexa
{
namespace sph
{
namespace cuda
{

template<class Dataset>
extern void computeDensity(size_t, size_t, size_t, Dataset& d, const cstone::Box<typename Dataset::RealType>&);

template<class Dataset>
extern void computeIAD(size_t, size_t, size_t, Dataset& d, const cstone::Box<typename Dataset::RealType>&);

template<class Dataset>
extern void computeMomentumAndEnergy(size_t, size_t, size_t, Dataset& d,
                                     const cstone::Box<typename Dataset::RealType>&);

} // namespace cuda
} // namespace sph
} // namespace sphexa
