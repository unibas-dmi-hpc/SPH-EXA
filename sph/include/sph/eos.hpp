#pragma once

#include <vector>

#include "kernels.hpp"

namespace sphexa
{
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
CUDA_DEVICE_HOST_FUN util::tuple<T2, T2> equationOfState(T1 u, T2 rho)
{
    constexpr T2 gamma = (5.0 / 3.0);

    T2 tmp = u * (gamma - T1(1));
    T2 p   = rho * tmp;
    T2 c   = std::sqrt(tmp);

    return {p, c};
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
CUDA_DEVICE_HOST_FUN util::tuple<T2, T2> equationOfState(T1 u, T2 rho, T3 mui)
{
    constexpr T2 R     = 8.317e7;
    constexpr T2 gamma = (5.0 / 3.0);

    T2 cv   = T2(1.5) * R / mui;
    T2 temp = u / cv;
    T2 tmp  = u * (gamma - T1(1));
    T2 p    = rho * tmp;
    T2 c    = std::sqrt(tmp);

    return {p, c, cv, temp};
}

/*! @brief Polytropic EOS
 *
 * @param rho  baryonic density
 *
 * Returns pressure, and speed of sound
 */
template<class T2>
CUDA_DEVICE_HOST_FUN util::tuple<T2, T2> equationOfState_Polytropic(T2 rho)
{
    constexpr T2 Kpol     = 2.246341237993810232e-10;
    constexpr T2 gammapol = 3.e0;

    T2 p    = Kpol * std::pow(rho, gammapol);
    T2 c    = std::sqrt(gammapol * p / rho);

    return {p, c};
}

/*! @brief ideal gas EOS interface for SPH where rho is computed on-the-fly
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
void computeEquationOfState(size_t startIndex, size_t endIndex, Dataset& d)
{
    const auto* u  = d.u.data();
    const auto* kx = d.kx.data();
    const auto* xm = d.xm.data();
    const auto* m  = d.m.data();

    auto* p = d.p.data();
    auto* c = d.c.data();

#pragma omp parallel for schedule(static)
    for (size_t i = startIndex; i < endIndex; ++i)
    {
        auto rho             = kx[i] * m[i] / xm[i];
        std::tie(p[i], c[i]) = equationOfState(u[i], rho);
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
void computeEquationOfState_Polytropic(size_t startIndex, size_t endIndex, Dataset& d)
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
        std::tie(p[i], c[i]) = equationOfState_Polytropic(rho);
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
void computeEquationOfState3L(size_t startIndex, size_t endIndex, Dataset& d)
{
    const auto* rho = d.rho.data();
    const auto* u   = d.u.data();

    auto* p = d.p.data();
    auto* c = d.c.data();

#pragma omp parallel for schedule(static)
    for (size_t i = startIndex; i < endIndex; ++i)
    {
        std::tie(p[i], c[i]) = equationOfState(u[i], rho[i]);
    }
}

} // namespace sph
} // namespace sphexa
