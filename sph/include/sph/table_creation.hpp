/*! @file
 * @brief Definition, selection and tabulation of smoothing kernels
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <cstdint>
#include <numeric>
#include <vector>

#include "kernels.hpp"

namespace util
{

/*! @brief integrate a function according to simpson's rule
 *
 * @tparam F
 * @param a      start of the integration interval
 * @param b      end of the integration interval
 * @param n      number of intervals
 * @param func   the integrand to be integrated
 * @return       the integral of func over [a, b]
 */
template<class F>
double simpson(double a, double b, uint64_t n, F&& func)
{
    uint64_t numOdd  = n / 2;
    uint64_t numEven = (numOdd >= 1) ? numOdd - 1 : 0;
    double   h       = (b - a) / double(n);

    std::vector<double> samplesOdd(numOdd);
    std::vector<double> samplesEven(numEven);

    for (uint64_t i = 0; i < numOdd; ++i)
    {
        uint64_t idx  = 2 * (i + 1) - 1;
        double   x    = a + double(idx) * h;
        samplesOdd[i] = func(x);
    }
    for (uint64_t i = 0; i < numEven; ++i)
    {
        uint64_t idx   = 2 * (i + 1);
        double   x     = a + double(idx) * h;
        samplesEven[i] = func(x);
    }
    // optional sorting for better accuracy
    std::sort(samplesOdd.begin(), samplesOdd.end());
    std::sort(samplesEven.begin(), samplesEven.end());

    return h / 3.0 *
           (func(a) + func(b) + 4.0 * std::accumulate(samplesOdd.begin(), samplesOdd.end(), 0.0) +
            2.0 * std::accumulate(samplesEven.begin(), samplesEven.end(), 0.0));
}

} // namespace util

namespace sph
{

//! @brief reference normalization constant from interpolation constants
template<typename T>
T sphynx_3D_k(T n)
{
    // b0, b1, b2 and b3 are defined in "SPHYNX: an accurate density-based SPH method for astrophysical applications",
    // DOI: 10.1051/0004-6361/201630208
    T b0 = 2.7012593e-2;
    T b1 = 2.0410827e-2;
    T b2 = 3.7451957e-3;
    T b3 = 4.7013839e-2;

    return b0 + b1 * std::sqrt(n) + b2 * n + b3 * std::sqrt(n * n * n);
}

//! @brief compute the 3D normalization constant for an arbitrary kernel
template<class F>
double kernel_3D_k(F&& sphKernel, double support)
{
    auto kernelVol3D = [sphKernel](double x) { return 4.0 * M_PI * x * x * sphKernel(x); };

    uint64_t numIntervals = 2000;
    return 1.0 / util::simpson(0, support, numIntervals, kernelVol3D);
}

//! @brief create a lookup-table for sinc(x)^sincIndex
template<typename T, std::size_t N>
std::array<T, N> createWharmonicTable(double sincIndex)
{
    constexpr int    numIntervals = N - 1;
    std::array<T, N> wh;

    constexpr T dx = 2.0 / numIntervals;
    for (size_t i = 0; i < N; ++i)
    {
        T normalizedVal = i * dx;
        wh[i]           = std::pow(wharmonic_std(normalizedVal), sincIndex);
    }
    return wh;
}

//! @brief create a lookup-table for d(sinc(x)^sincIndex)/dx
template<typename T, std::size_t N>
std::array<T, N> createWharmonicDerivativeTable(double sincIndex)
{
    constexpr int    numIntervals = N - 1;
    std::array<T, N> whd;

    const T dx = 2.0 / numIntervals;
    for (size_t i = 0; i < N; ++i)
    {
        T normalizedVal = i * dx;
        whd[i] =
            sincIndex * std::pow(wharmonic_std(normalizedVal), sincIndex - 1) * wharmonic_derivative_std(normalizedVal);
    }

    return whd;
}

} // namespace sph