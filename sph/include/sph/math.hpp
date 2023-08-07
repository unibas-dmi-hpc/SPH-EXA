#pragma once

#include <cmath>

#include "cstone/cuda/annotation.hpp"
#include "cstone/util/array.hpp"

namespace sph
{

//! @brief symmetric 3x3 matrix-vector product
template<class Tv, class Tm>
HOST_DEVICE_FUN HOST_DEVICE_INLINE util::array<Tv, 3> symv(const util::array<Tm, 6>& mat, const util::array<Tv, 3>& vec)
{
    util::array<Tv, 3> ret;
    ret[0] = mat[0] * vec[0] + mat[1] * vec[1] + mat[2] * vec[2];
    ret[1] = mat[3] * vec[1] + mat[4] * vec[2];
    ret[2] = mat[5] * vec[2];
    return ret;
}

} // namespace sph
