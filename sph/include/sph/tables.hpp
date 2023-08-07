#pragma once

#include "cstone/cuda/annotation.hpp"
#include "kernels.hpp"

namespace sph
{
namespace lt
{

constexpr size_t size = 20000;

//! @brief create a lookup-table for sinc(x)^sincIndex
template<typename T, std::size_t N>
std::array<T, N> createWharmonicLookupTable(T sincIndex)
{
    std::array<T, N> wh;

    const T halfsSize = N / 2.0;
    for (size_t i = 0; i < N; ++i)
    {
        T normalizedVal = i / halfsSize;
        wh[i]           = std::pow(wharmonic_std(normalizedVal), sincIndex);
    }
    return wh;
}

//! @brief create a lookup-table for d(sinc(x)^sincIndex)/dx
template<typename T, std::size_t N>
std::array<T, N> createWharmonicDerivativeLookupTable(T sincIndex)
{
    std::array<T, N> whd;

    const T halfsSize = N / 2.0;
    for (size_t i = 0; i < N; ++i)
    {
        T normalizedVal = i / halfsSize;
        whd[i] =
            sincIndex * std::pow(wharmonic_std(normalizedVal), sincIndex - 1) * wharmonic_derivative_std(normalizedVal);
    }

    return whd;
}

template<typename T>
HOST_DEVICE_FUN inline T wharmonic_lt_with_derivative(const T* wh, const T* whd, T v)
{
    constexpr size_t halfTableSize   = size / 2;
    constexpr T      inverseHalfSize = T(1) / halfTableSize;

    const size_t idx = v * halfTableSize;
    return (idx >= size) ? 0.0 : wh[idx] + whd[idx] * (v - T(idx) * inverseHalfSize);
}

} // namespace lt
} // namespace sph
