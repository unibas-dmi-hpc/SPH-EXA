#pragma once

#include <vector>

#include "kernels.hpp"

namespace sph
{

/*! @brief Reduced version of Ideal gas EOS for internal energy
 *
 * @param u    internal energy
 * @param rho  baryonic density
 *
 * This EOS is used for simple cases where we don't need the temperature.
 * Returns pressure, speed of sound
 */
template<class T1, class T2>
CUDA_DEVICE_HOST_FUN auto idealGasEOS(T1 u, T2 rho)
{
    using Tc           = std::common_type_t<T1, T2>;
    constexpr Tc gamma = (5.0 / 3.0);

    Tc tmp = u * (gamma - Tc(1));
    Tc p   = rho * tmp;
    Tc c   = std::sqrt(tmp);

    return util::tuple<Tc, Tc>{p, c};
}

/*! @brief Ideal gas EOS for internal energy taking into account composition via mui
 *
 * @param u    internal energy
 * @param rho  baryonic density
 * @param mui  mean molecular weight
 *
 * Returns pressure, speed of sound, du/dT, and temperature
 */
template<class T1, class T2, class T3>
CUDA_DEVICE_HOST_FUN auto idealGasEOS(T1 u, T2 rho, T3 mui)
{
    using Tc = std::common_type_t<T1, T2, T3>;

    constexpr Tc R     = 8.317e7;
    constexpr Tc gamma = (5.0 / 3.0);

    Tc cv   = Tc(1.5) * R / mui;
    Tc temp = u / cv;
    Tc tmp  = u * (gamma - Tc(1));
    Tc p    = rho * tmp;
    Tc c    = std::sqrt(tmp);

    return util::tuple<Tc, Tc, Tc, Tc>{p, c, cv, temp};
}

/*! @brief Polytropic EOS for a 1.4 M_sun and 12.8 km neutron star
 *
 * @param rho  baryonic density
 *
 * Kpol is hardcoded for these NS characteristics and is not valid for
 * other NS masses and radius
 * Returns pressure, and speed of sound
 */
template<class T>
CUDA_DEVICE_HOST_FUN auto polytropicEOS(T rho)
{
    constexpr T Kpol     = 2.246341237993810232e-10;
    constexpr T gammapol = 3.e0;

    T p = Kpol * std::pow(rho, gammapol);
    T c = std::sqrt(gammapol * p / rho);

    return util::tuple<T, T>{p, c};
}

/*! @brief ideal gas EOS interface w/o temperature for SPH where rho is computed on-the-fly
 *
 * @tparam Dataset
 * @param startIndex  index of first locally owned particle
 * @param endIndex    index of last locally owned particle
 * @param d           the dataset with the particle buffers
 *
 * In this simple version of equation of state, we calculate all dependent quantities
 * also for halos, not just assigned particles in [startIndex:endIndex], so that
 * we could potentially avoid halo exchange of p and c in return for exchanging halos of u.
 */
template<typename Dataset>
void computeEOS(size_t startIndex, size_t endIndex, Dataset& d)
{
    const auto* u     = d.u.data();
    const auto* m     = d.m.data();
    const auto* kx    = d.kx.data();
    const auto* xm    = d.xm.data();
    const auto* gradh = d.gradh.data();

    auto* p    = d.p.data();
    auto* prho = d.prho.data();
    auto* c    = d.c.data();

    bool storeRho = (d.rho.size() == d.m.size());

#pragma omp parallel for schedule(static)
    for (size_t i = startIndex; i < endIndex; ++i)
    {
        auto rho             = kx[i] * m[i] / xm[i];
        std::tie(p[i], c[i]) = idealGasEOS(u[i], rho);
        prho[i]              = p[i] / (kx[i] * m[i] * m[i] * gradh[i]);
        if (storeRho) { d.rho[i] = rho; }
    }
}

/*! @brief Polytropic EOS interface for SPH where rho is computed on-the-fly
 *
 * @tparam Dataset
 * @param startIndex  index of first locally owned particle
 * @param endIndex    index of last locally owned particle
 * @param d           the dataset with the particle buffers
 */
template<typename Dataset>
void computeEOS_Polytropic(size_t startIndex, size_t endIndex, Dataset& d)
{
    const auto* kx = d.kx.data();
    const auto* xm = d.xm.data();
    const auto* m  = d.m.data();

    auto* p = d.p.data();
    auto* c = d.c.data();

#pragma omp parallel for schedule(static)
    for (size_t i = startIndex; i < endIndex; ++i)
    {
        auto rho             = kx[i] * m[i] / xm[i];
        std::tie(p[i], c[i]) = polytropicEOS(rho);
    }
}

/*! @brief Ideal gas EOS interface w/o temperature for SPH where rho is stored
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
void computeEOS_HydroStd(size_t startIndex, size_t endIndex, Dataset& d)
{
    const auto* rho = d.rho.data();
    const auto* u   = d.u.data();

    auto* p = d.p.data();
    auto* c = d.c.data();

#pragma omp parallel for schedule(static)
    for (size_t i = startIndex; i < endIndex; ++i)
    {
        std::tie(p[i], c[i]) = idealGasEOS(u[i], rho[i]);
    }
}

} // namespace sph
