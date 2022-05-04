#pragma once

#include <vector>

#include "kernels.hpp"

namespace sphexa
{
namespace sph
{

/*! @brief compute temperature, pressure and heat capacity from density and internal energy
 *
 * @tparam Dataset
 * @param startIndex  index of first locally owned particle
 * @param endIndex    index of last locally owned particle
 * @param d           the dataset with the particle buffers
 *
 * In this simple version of state equation, we calculate all depended quantities
 * also for halos, not just assigned particles in [startIndex:endIndex], so that
 * we can avoid halo exchange of p and c.
 */
template<typename Dataset>
void computeEquationOfState(size_t /*startIndex*/, size_t /*endIndex*/, Dataset& d)
{
    using T = typename Dataset::RealType;

    // const T R     = 8.317e7;
    const T gamma = (5.0 / 3.0);

    const T* rho = d.rho.data();
    const T* u   = d.u.data();
    // const T* mui = d.mui.data();

    const size_t numParticles = d.x.size();

    // T* cv   = d.cv.data();
    // T* temp = d.temp.data();
    T* p = d.p.data();
    T* c = d.c.data();

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numParticles; ++i)
    {
        // cv[i]   = T(1.5) * R / mui[i];
        // temp[i] = u[i] / cv[i];
        T tmp = u[i] * (gamma - 1);
        p[i]  = rho[i] * tmp;
        c[i]  = std::sqrt(tmp);
    }
}

//! @brief same as computeEquationOfState, but compute as rho = kx * rho0
template<typename Dataset>
void computeEquationOfStateVE(size_t /*startIndex*/, size_t /*endIndex*/, Dataset& d)
{
    using T = typename Dataset::RealType;

    // const T R     = 8.317e7;
    const T gamma = (5.0 / 3.0);

    const T* kx   = d.kx.data();
    const T* rho0 = d.rho0.data();
    const T* u    = d.u.data();
    // const T* mui = d.mui.data();

    const size_t numParticles = d.x.size();

    // T* cv   = d.cv.data();
    // T* temp = d.temp.data();
    T* p = d.p.data();
    T* c = d.c.data();

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numParticles; ++i)
    {
        // cv[i]   = T(1.5) * R / mui[i];
        // temp[i] = u[i] / cv[i];
        T tmp = u[i] * (gamma - 1);
        p[i]  = kx[i] * rho0[i] * tmp;
        c[i]  = std::sqrt(tmp);
    }
}

} // namespace sph
} // namespace sphexa
