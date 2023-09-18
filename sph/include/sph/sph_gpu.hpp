#pragma once

#include "cstone/sfc/box.hpp"
#include "cstone/tree/octree.hpp"

namespace sph
{

template<class Dataset>
extern void computeTargetGroups(size_t, size_t, Dataset& d, const cstone::Box<typename Dataset::RealType>&);

template<class Dataset>
extern void computeDensityGpu(size_t, size_t, Dataset& d, const cstone::Box<typename Dataset::RealType>&);

template<class Dataset>
extern void computeIADGpu(size_t, size_t, Dataset& d, const cstone::Box<typename Dataset::RealType>&);

template<class Dataset>
extern void computeMomentumEnergyStdGpu(size_t, size_t, Dataset& d, const cstone::Box<typename Dataset::RealType>&);

namespace cuda
{

template<class Dataset>
extern void computeXMass(size_t, size_t, Dataset& d, const cstone::Box<typename Dataset::RealType>&);

template<class Dataset>
extern void computeVeDefGradh(size_t, size_t, Dataset& d, const cstone::Box<typename Dataset::RealType>&);

template<class Dataset>
extern void computeIadDivvCurlv(size_t, size_t, Dataset& d, const cstone::Box<typename Dataset::RealType>&);

template<class Dataset>
extern void computeAVswitches(size_t, size_t, Dataset& d, const cstone::Box<typename Dataset::RealType>&);

template<bool avClean, class Dataset>
extern void computeMomentumEnergy(size_t, size_t, Dataset& d, const cstone::Box<typename Dataset::RealType>&);

template<class Tu, class Trho, class Tp, class Tc>
extern void computeEOS_HydroStd(size_t, size_t, Trho, Tu, const Tu*, const Trho*, Tp*, Tc*);

template<class Tu, class Tm, class Thydro>
extern void computeEOS(size_t, size_t, Tm mui, Tu gamma, const Tu*, const Tm*, const Thydro*, const Thydro*,
                       const Thydro*, Thydro*, Thydro*, Thydro*, Thydro*);

} // namespace cuda

template<class Tc, class Tv, class Ta, class Tdu, class Tm1, class Tu, class Thydro>
extern void computePositionsGpu(size_t first, size_t last, double dt, double dt_m1, Tc* x, Tc* y, Tc* z, Tv* vx, Tv* vy,
                                Tv* vz, Tm1* x_m1, Tm1* y_m1, Tm1* z_m1, Ta* ax, Ta* ay, Ta* az, Tu* temp, Tu* u,
                                Tdu* du, Tm1* du_m1, Thydro* h, Thydro* mui, Tc gamma, Tc constCv,
                                const cstone::Box<Tc>& box);

template<class Th>
extern void updateSmoothingLengthGpu(size_t, size_t, unsigned ng0, const unsigned* nc, Th* h);

} // namespace sph
