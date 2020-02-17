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
    // 13.2.2020: this is most likely wrong. (missing kernel index multiplication (ruben has the derivation of this)).
    if (v == 0.0) return 0.0;

    const T Pv = (PI / 2.0) * v;
    const T sincv = std::sin(Pv) / (Pv); // This is missing a multiplicationi with the kernel index (see below)

    return sincv * (PI / 2.0) * ((std::cos(Pv) / std::sin(Pv)) - 1.0 / Pv); // this is used in lookuptable or not.
    //kernelindex * sincv * v / h**4[i] * K * (PI / 2.0) * ((std::cos(Pv) / std::sin(Pv)) - 1.0 / Pv); // h**4[i] needs to go outside (obviously...). compare to sphynx lookup table
    // second column of sphynx LT is pi/2(1/tan((pi/2)*v) - 1/((pi/2)*v)) * v  [sphynx doesn't have the * v, for cleanness (multiplied when using the derivative), but it could be here for saving operations]
    // generally, the second column of the lookuptable should be the ratio of w'/w so that we can reconstruct w' by multiplying the second column with the kernel (w' = w'/w * w)...
    // additionally, don't forget the K * kernelIndex / h[i] (see picture of ruben's ntes towards the center of the page.. in the notes, the h is included, but we won't do that in the code, because it depends on [i]
    // there is a trick how to get the kernelIndex-1 from the derivative back to the same (kernelIndex in the calculation)
    // also, the sincv MUST NOT BE in the derivative lookup-table, because it couples the lookuptable to the index... (i.e. the version here is not correct! but as it looks it is not yet used anyways)
}

template <typename T>
CUDA_DEVICE_FUN inline T artificial_viscosity(const T ro_i, const T ro_j, const T h_i, const T h_j, const T c_i, const T c_j, const T rv,
                                              const T r_square)
{
    const T alpha = 1.0;
    const T beta = 2.0;
    const T epsilon = 0.01;

    const T ro_ij = (ro_i + ro_j) / 2.0;
    const T c_ij = (c_i + c_j) / 2.0;
    const T h_ij = (h_i + h_j) / 2.0;

    // calculate viscosity_ij according to Monaghan & Gringold 1983
    T viscosity_ij = 0.0;
    if (rv < 0.0)
    {
        // calculate muij
        const T mu_ij = (h_ij * rv) / (r_square + epsilon * h_ij * h_ij);
        viscosity_ij = (-alpha * c_ij * mu_ij + beta * mu_ij * mu_ij) / ro_ij;
    }

    return viscosity_ij;
}

} // namespace sphexa
