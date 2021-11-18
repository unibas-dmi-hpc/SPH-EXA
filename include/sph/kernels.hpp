#pragma once

#include "math.hpp"
#include "cudaFunctionAnnotation.hpp"

namespace sphexa
{

template <typename T>
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

template <typename T>
CUDA_DEVICE_HOST_FUN inline T wharmonic_std(T v)
{
    if (v == 0.0) return 1.0;

    const T Pv = (PI / 2.0) * v;

    return std::sin(Pv) / Pv;
}

template <typename T>
CUDA_DEVICE_HOST_FUN inline T wharmonic_derivative_std(T v)
{
    if (v == 0.0) return 0.0;

    const T Pv = (PI / 2.0) * v;
    const T sincv = std::sin(Pv) / (Pv);

    return sincv * (PI / 2.0) * ((std::cos(Pv) / std::sin(Pv)) - 1.0 / Pv);
}

template<typename T>
CUDA_DEVICE_FUN inline T artificial_viscosity(T ro_i, T ro_j, T h_i, T h_j, T c_i, T c_j, T rv, T r_square)
{
    constexpr T alpha   = 1.0;
    constexpr T beta    = 2.0;
    constexpr T epsilon = 0.01;

    T ro_ij = (ro_i + ro_j) * T(0.5);
    T c_ij  = (c_i + c_j) * T(0.5);
    T h_ij  = (h_i + h_j) * T(0.5);

    // calculate viscosity_ij according to Monaghan & Gringold 1983
    T viscosity_ij = 0.0;
    if (rv < 0.0)
    {
        // calculate muij
        T mu_ij = (h_ij * rv) / (r_square + epsilon * h_ij * h_ij);
        viscosity_ij  = (-alpha * c_ij * mu_ij + beta * mu_ij * mu_ij) / ro_ij;
    }

    return viscosity_ij;
}

template<typename T>
CUDA_DEVICE_FUN inline T artificial_viscosity_sphynx(T ro_i, T ro_j, T h_i, T h_j, T c_i, T c_j, T rv,
                                                     T r_square)
{
    constexpr T alpha   = 1.0;
    constexpr T beta    = 2.0;
    constexpr T epsilon = 0.01;

    //T ro_ij = (ro_i + ro_j) * T(0.5);
    //T c_ij  = (c_i + c_j) * T(0.5);
    //T h_ij  = (h_i + h_j) * T(0.5);

    // calculate viscosity_ij according to Monaghan & Gringold 1983
    T viscosity_ij = 0.0;
    if (rv < 0.0)
    {
        T wij = rv / std::sqrt(r_square);
        T vij_signal = (alpha + alpha) / 4 * (c_i + c_j) - beta * wij;
        viscosity_ij  = -vij_signal * wij;
    }

    return viscosity_ij;
}

} // namespace sphexa
