#pragma once

#include "math.hpp"
#include "sph/util/annotation.hpp"

namespace sph
{

template<typename T>
CUDA_DEVICE_HOST_FUN inline T compute_3d_k(T n)
{
    // b0, b1, b2 and b3 are defined in "SPHYNX: an accurate density-based SPH method for astrophysical applications",
    // DOI: 10.1051/0004-6361/201630208
    T b0 = 2.7012593e-2;
    T b1 = 2.0410827e-2;
    T b2 = 3.7451957e-3;
    T b3 = 4.7013839e-2;

    return b0 + b1 * std::sqrt(n) + b2 * n + b3 * std::sqrt(n * n * n);
}

//! @brief compute time-step based on the signal velocity
template<class T1, class T2, class T3>
CUDA_DEVICE_HOST_FUN inline auto tsKCourant(T1 maxvsignal, T2 h, T3 c, double kcour)
{
    using T = std::common_type_t<T1, T2, T3>;
    T v     = maxvsignal > T(0) ? maxvsignal : c;
    return T(kcour * h / v);
}

//! @brief sinc(PI/2 * v)
template<typename T>
CUDA_DEVICE_HOST_FUN inline T wharmonic_std(T v)
{
    if (v == 0.0) return 1.0;

    const T Pv = (PI / 2.0) * v;

    return std::sin(Pv) / Pv;
}

/*! @brief Derivative of sinc(PI/2 * v) w.r to v
 *
 * Unoptimized for clarity as this is only used to construct look-up tables
 */
template<typename T>
CUDA_DEVICE_HOST_FUN inline T wharmonic_derivative_std(T v)
{
    if (v == 0.0) return 0.0;

    constexpr T piHalf = PI / 2.0;

    const T Pv    = piHalf * v;
    const T sincv = std::sin(Pv) / (Pv);

    return sincv * piHalf * ((std::cos(Pv) / std::sin(Pv)) - T(1) / Pv);
}

template<typename T>
CUDA_DEVICE_HOST_FUN inline T wharmonic_derivative(T v, T powsincv)
{
    if (v == T(0)) return T(0);

    constexpr T piHalf = PI / 2.0;

    const T Pv = piHalf * v;
    return powsincv * piHalf * (T(1) / std::tan(Pv) - T(1) / Pv);
}

/*! @brief Old viscosity according to Monaghan & Gringold 1983
 *
 * We found that this leads to way too much noise in the radial velocity and radial pressure gradients
 * in the Evrard collapse test case in the wake of the shock wave.
 */
template<typename T>
CUDA_DEVICE_HOST_FUN inline T artificial_viscosity_old(T ro_i, T ro_j, T h_i, T h_j, T c_i, T c_j, T rv, T r_square)
{
    constexpr T alpha   = 1.0;
    constexpr T beta    = 2.0;
    constexpr T epsilon = 0.01;

    T ro_ij = (ro_i + ro_j) * T(0.5);
    T c_ij  = (c_i + c_j) * T(0.5);
    T h_ij  = (h_i + h_j) * T(0.5);

    T viscosity_ij = 0.0;
    if (rv < 0.0)
    {
        T mu_ij      = (h_ij * rv) / (r_square + epsilon * h_ij * h_ij);
        viscosity_ij = (-alpha * c_ij * mu_ij + beta * mu_ij * mu_ij) / ro_ij;
    }

    return viscosity_ij;
}

/*! @brief calculate the artificial viscosity between a pair of two particles
 *
 * @tparam T      float or double
 * @param alpha_i viscosity switch of particle i
 * @param alpha_j viscosity switch of particle j
 * @param c_i     speed of sound particle i
 * @param c_j     speed of sound particle j
 * @param w_ij    relative velocity (v_i - v_j), projected onto the connecting axis (r_i - r_j)
 * @return        the viscosity
 */
template<typename T>
CUDA_DEVICE_HOST_FUN inline T artificial_viscosity(T alpha_i, T alpha_j, T c_i, T c_j, T w_ij)
{
    // alpha is const for now, but will be different for each particle when using viscosity switching
    constexpr T beta = 2.0;

    T viscosity_ij = 0.0;
    if (w_ij < 0.0)
    {
        T vij_signal = (alpha_i + alpha_j) / 4.0 * (c_i + c_j) - beta * w_ij;
        viscosity_ij = -vij_signal * w_ij;
    }

    return viscosity_ij;
}

} // namespace sph
