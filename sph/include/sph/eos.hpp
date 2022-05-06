#pragma once

#include <vector>

#include "kernels.hpp"

namespace sphexa
{
namespace sph
{

/*! @brief Ideal gas EOS for SPH formulations where rho is computed on-the-fly
 *
 * @tparam Dataset
 * @param startIndex  index of first locally owned particle
 * @param endIndex    index of last locally owned particle
 * @param d           the dataset with the particle buffers
 *
 * In this simple version of state equation, we calculate all depended quantities
 * also for halos, not just assigned particles in [startIndex:endIndex], so that
 * we could potentially avoid halo exchange of p and c in return for exchanging halos of u.
 */
template<typename Dataset>
void computeEquationOfState(size_t /*startIndex*/, size_t /*endIndex*/, Dataset& d)
{
    using T = std::decay_t<decltype(d.u[0])>;

    // const T R     = 8.317e7;
    const T gamma = (5.0 / 3.0);

    const auto* u  = d.u.data();
    const auto* kx = d.kx.data();
    const auto* xm = d.xm.data();
    const auto* m  = d.m.data();
    // const auto* mui = d.mui.data();

    const size_t numParticles = d.x.size();

    // auto* cv   = d.cv.data();
    // auto* temp = d.temp.data();
    auto* p = d.p.data();
    auto* c = d.c.data();

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numParticles; ++i)
    {
        // cv[i]   = T(1.5) * R / mui[i];
        // temp[i] = u[i] / cv[i];
        T tmp = u[i] * (gamma - T(1));
        T rho = kx[i] * m[i] / xm[i];
        p[i]  = rho * tmp;
        c[i]  = std::sqrt(tmp);
    }
}

/*! @brief Ideal gas EOS for SPH formulations where rho is stored
 *
 * @tparam Dataset
 * @param startIndex  index of first locally owned particle
 * @param endIndex    index of last locally owned particle
 * @param d           the dataset with the particle buffers
 *
 * In this simple version of state equation, we calculate all depended quantities
 * also for halos, not just assigned particles in [startIndex:endIndex], so that
 * we could potentially avoid halo exchange of p and c in return for exchanging halos of u.
 */
template<typename Dataset>
void computeEquationOfState3L(size_t /*startIndex*/, size_t /*endIndex*/, Dataset& d)
{
    using T = std::decay_t<decltype(d.u[0])>;

    // const T R     = 8.317e7;
    const T gamma = (5.0 / 3.0);

    const auto* rho = d.rho.data();
    const auto* u   = d.u.data();
    // const auto* mui = d.mui.data();

    const size_t numParticles = d.x.size();

    // auto* cv   = d.cv.data();
    // auto* temp = d.temp.data();
    auto* p = d.p.data();
    auto* c = d.c.data();

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numParticles; ++i)
    {
        // cv[i]   = T(1.5) * R / mui[i];
        // temp[i] = u[i] / cv[i];
        T tmp = u[i] * (gamma - T(1));
        p[i]  = rho[i] * tmp;
        c[i]  = std::sqrt(tmp);
    }
}

} // namespace sph
} // namespace sphexa
