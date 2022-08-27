#pragma once

#include "cstone/sfc/box.hpp"

//! @brief maximum number of neighbors supported in GPU kernels
#define NGMAX 150

namespace sph
{
namespace cuda
{

template<class Dataset>
extern void computeDensity(size_t, size_t, int, Dataset& d, const cstone::Box<typename Dataset::RealType>&);

template<class Dataset>
extern void computeIAD(size_t, size_t, int, Dataset& d, const cstone::Box<typename Dataset::RealType>&);

template<class Dataset>
extern void computeMomentumEnergySTD(size_t, size_t, int, Dataset& d, const cstone::Box<typename Dataset::RealType>&);

template<class Dataset>
extern void computeXMass(size_t, size_t, int, Dataset& d, const cstone::Box<typename Dataset::RealType>&);

template<class Dataset>
extern void computeVeDefGradh(size_t, size_t, int, Dataset& d, const cstone::Box<typename Dataset::RealType>&);

template<class Dataset>
extern void computeIadDivvCurlv(size_t, size_t, int, Dataset& d, const cstone::Box<typename Dataset::RealType>&);

template<class Dataset>
extern void computeAVswitches(size_t, size_t, int, Dataset& d, const cstone::Box<typename Dataset::RealType>&);

template<class Dataset>
extern void computeMomentumEnergy(size_t, size_t, int, Dataset& d, const cstone::Box<typename Dataset::RealType>&);

template<class Tu, class Trho, class Tp, class Tc>
extern void computeEOS_HydroStd(size_t, size_t, Tu, const Tu*, const Trho*, Tp*, Tc*);

template<class Tu, class Tm, class Thydro>
extern void computeEOS(size_t, size_t, Tu, const Tu*, const Tm*, const Thydro*, const Thydro*, const Thydro*, Thydro*,
                       Thydro*);

} // namespace cuda
} // namespace sph
