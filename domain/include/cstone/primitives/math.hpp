/*! @file
 * @brief  Basic math
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <cstdint>

#include "cstone/cuda/annotation.hpp"

namespace cstone
{

//! @brief ceil(dividend/divisor) for unsigned integers
HOST_DEVICE_FUN constexpr size_t iceil(size_t dividend, unsigned divisor)
{
    return (dividend + divisor - 1lu) / divisor;
}

//! @brief round up @p n to multiple of @p m
HOST_DEVICE_FUN constexpr size_t round_up(size_t n, unsigned m) { return ((n + m - 1) / m) * m; }

} // namespace cstone