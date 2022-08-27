#pragma once

#include <cmath>

#include <cstdio>
#include <array>

#include "cstone/cuda/annotation.hpp"

#define PI 3.14159265358979323846

namespace sph
{
namespace math
{

template<typename T>
HOST_DEVICE_FUN inline T pow(T a, int b)
{
    if (b == 0)
        return 1;
    else if (b == 1)
        return a;
    else if (b == 2)
        return a * a;
    else if (b == 3)
        return a * a * a;
    else if (b == 4)
        return a * a * a * a;
    else if (b == 5)
        return a * a * a * a * a;
    else if (b == 6)
        return a * a * a * a * a * a;
    else if (b == 7)
        return a * a * a * a * a * a * a;
    else
        return std::pow(a, b);
}

} // namespace math
} // namespace sph
