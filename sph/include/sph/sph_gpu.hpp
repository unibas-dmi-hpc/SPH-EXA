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

template<class Tc, class Tv, class Tm1, class Tu, class Thydro>
extern void computePositionsGpu(size_t, size_t, double, double, Tc* x, Tc* y, Tc* z, Tv* vx, Tv* vy, Tv* vz, Tm1* x_m1,
                                Tm1* y_m1, Tm1* z_m1, Tu* u, Tm1* du, Tm1* du_m1, Thydro* h, const cstone::Box<Tc>&);

} // namespace cuda
} // namespace sph
