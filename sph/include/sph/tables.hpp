#pragma once

#include "sph/util/annotation.hpp"
#include "kernels.hpp"

namespace sph
{
namespace lt
{

constexpr size_t size = 20000;

template<typename T, std::size_t N>
std::array<T, N> createWharmonicLookupTable()
{
    std::array<T, N> wh;

    const T halfsSize = N / 2.0;
    for (size_t i = 0; i < N; ++i)
    {
        T normalizedVal = i / halfsSize;
        wh[i]           = wharmonic_std(normalizedVal);
    }
    return wh;
}

template<typename T, std::size_t N>
std::array<T, N> createWharmonicDerivativeLookupTable()
{
    std::array<T, N> whd;

    const T halfsSize = N / 2.0;
    for (size_t i = 0; i < N; ++i)
    {
        T normalizedVal = i / halfsSize;
        whd[i]          = wharmonic_derivative_std(normalizedVal);
    }

    return whd;
}

// template <typename T>
// CUDA_DEVICE_FUN inline T wharmonic_lt(const T *wh, const size_t ltsize, const T v)
// {
//     const size_t idx = (v * ltsize / 2.0);
//     return (idx >= ltsize) ? 0.0 : wh[idx];
// }

template<typename T>
CUDA_DEVICE_HOST_FUN inline T wharmonic_lt_with_derivative(const T* wh, const T* whd, T v)
{
    constexpr size_t halfTableSize   = size / 2;
    constexpr double inverseHalfSize = 1.0 / halfTableSize;

    const size_t idx = v * halfTableSize;
    return (idx >= size) ? 0.0 : wh[idx] + whd[idx] * (v - T(idx) * inverseHalfSize);
}

// template <typename T>
// CUDA_DEVICE_FUN inline T wharmonic_derivative_lt(const T *whd, const size_t ltsize, const T v)
// {
//     const size_t idx = (v * ltsize / 2.0);
//     return (idx >= ltsize) ? -0.5 : whd[idx];
// }

} // namespace lt

} // namespace sph
