#pragma once

#include "cstone/cuda/annotation.hpp"

namespace sph
{
namespace lt
{

constexpr int kTableSize = 20000;

//! @brief lookup values of a tabulated function with linear interpolation
template<typename T>
HOST_DEVICE_FUN inline T lookup(const T* table, T v)
{
    constexpr int numIntervals = kTableSize - 1;
    constexpr T   support      = 2.0;
    constexpr T   dx           = support / numIntervals;
    constexpr T   invDx        = T(1) / dx;

    int idx = v * invDx;

    T derivative = (idx >= numIntervals) ? 0.0 : (table[idx + 1] - table[idx]) * invDx;

    return (idx >= numIntervals) ? 0.0 : table[idx] + derivative * (v - T(idx) * dx);
}

} // namespace lt
} // namespace sph
