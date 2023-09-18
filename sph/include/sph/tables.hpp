#pragma once

#include "cstone/cuda/annotation.hpp"
#include "kernels.hpp"

namespace sph
{
namespace lt
{

constexpr int size = 20000;

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

//! @brief lookup values of a tabulated function with linear interpolation
template<typename T>
HOST_DEVICE_FUN inline T lookup(const T* table, T v)
{
    constexpr int numIntervals = size - 1;
    constexpr T   support      = 2.0;
    constexpr T   dx           = support / numIntervals;
    constexpr T   invDx        = T(1) / dx;

    int idx = v * invDx;

    T derivative = (idx >= numIntervals) ? 0.0 : (table[idx + 1] - table[idx]) * invDx;

    return (idx >= numIntervals) ? 0.0 : table[idx] + derivative * (v - T(idx) * dx);
}

} // namespace lt
} // namespace sph
