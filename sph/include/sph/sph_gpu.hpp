#pragma once

#include "cstone/sfc/box.hpp"
#include "cstone/tree/octree.hpp"
#include "sph/groups.hpp"
#include "cstone/tree/definitions.h"

namespace sph
{

template<class Dataset>
extern void computeSpatialGroups(size_t, size_t, Dataset& d, const cstone::Box<typename Dataset::RealType>&,
                                 GroupData<cstone::GpuTag>&);

template<class Dataset>
extern void computeIADGpu(const GroupView&, Dataset& d, const cstone::Box<typename Dataset::RealType>&);

template<class Dataset>
extern void computeMomentumEnergyStdGpu(const GroupView&, Dataset& d, const cstone::Box<typename Dataset::RealType>&);

namespace cuda
{

template<class Dataset>
extern void computeXMass(const GroupView&, Dataset& d, const cstone::Box<typename Dataset::RealType>&);

template<class Dataset>
void computeDensity(const GroupView&, Dataset& d, const cstone::Box<typename Dataset::RealType>& box);

template<class Dataset>
extern void computeVeDefGradh(const GroupView&, Dataset& d, const cstone::Box<typename Dataset::RealType>&);

template<class Dataset>
extern void computeIadDivvCurlv(const GroupView&, Dataset& d, const cstone::Box<typename Dataset::RealType>&);

template<class Dataset>
extern void computeAVswitches(const GroupView&, Dataset& d, const cstone::Box<typename Dataset::RealType>&);

template<bool avClean, class Dataset>
extern void computeMomentumEnergy(const GroupView&, float*, Dataset&, const cstone::Box<typename Dataset::RealType>&);

template<class Tu, class Trho, class Tp, class Tc>
extern void computeEOS_HydroStd(size_t, size_t, Trho, Tu, const Tu*, const Trho* m, Trho*, Tp*, Tc*);

template<class Tu, class Tm, class Thydro>
extern void computeEOS(size_t, size_t, Tm mui, Tu gamma, const Tu*, const Tm*, const Thydro*, const Thydro*,
                       const Thydro*, Thydro*, Thydro*, Thydro*, Thydro*);

} // namespace cuda

template<class Tc, class Thydro, class Tm1>
extern void driftPositions(GroupView grp, float dt, float dt_back, float dt_m1, Tc* x, Tc* y, Tc* z, Thydro* vx,
                           Thydro* vy, Thydro* vz, const Tm1* x_m1, const Tm1* y_m1, const Tm1* z_m1, const Thydro* ax,
                           const Thydro* ay, const Thydro* az);

template<class Tc, class Tv, class Ta, class Tdu, class Tm1, class Tu, class Thydro>
extern void computePositionsGpu(size_t first, size_t last, double dt, double dt_m1, Tc* x, Tc* y, Tc* z, Tv* vx, Tv* vy,
                                Tv* vz, Tm1* x_m1, Tm1* y_m1, Tm1* z_m1, Ta* ax, Ta* ay, Ta* az, Tu* temp, Tu* u,
                                Tdu* du, Tm1* du_m1, Thydro* h, Thydro* mui, Tc gamma, Tc constCv,
                                const cstone::Box<Tc>& box);

template<class Th>
extern void updateSmoothingLengthGpu(size_t, size_t, unsigned ng0, const unsigned* nc, Th* h);

template<class T>
extern void groupDivvTimestepGpu(float Krho, const GroupView&, const T* divv, float* groupDt);

template<class T>
extern void groupAccTimestepGpu(float etaAcc, const GroupView&, const T* ax, const T* ay, const T* az, float* groupDt);

} // namespace sph
