#pragma once

#include "cstone/sfc/box.hpp"

//! @brief maximum number of neighbors supported in GPU kernels
#define NGMAX 150

namespace sph
{
namespace cuda
{

template<class Dataset>
extern void computeDensity(size_t, size_t, size_t, Dataset& d, const cstone::Box<typename Dataset::RealType>&);

template<class Dataset>
extern void computeIAD(size_t, size_t, size_t, Dataset& d, const cstone::Box<typename Dataset::RealType>&);

template<class Dataset>
extern void computeMomentumEnergySTD(size_t, size_t, size_t, Dataset& d,
                                     const cstone::Box<typename Dataset::RealType>&);

template<class Dataset>
extern void computeXMass(size_t, size_t, size_t, Dataset& d, const cstone::Box<typename Dataset::RealType>&);

template<class Dataset>
extern void computeVeDefGradh(size_t, size_t, size_t, Dataset& d, const cstone::Box<typename Dataset::RealType>&);

template<class Dataset>
extern void computeIadDivvCurlv(size_t, size_t, size_t, Dataset& d, const cstone::Box<typename Dataset::RealType>&);

template<class Dataset>
extern void computeAVswitches(size_t, size_t, size_t, Dataset& d, const cstone::Box<typename Dataset::RealType>&);

template<class Dataset>
extern void computeMomentumEnergy(size_t, size_t, size_t, Dataset& d, const cstone::Box<typename Dataset::RealType>&);

} // namespace cuda
} // namespace sph
