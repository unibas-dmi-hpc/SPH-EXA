/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief implements a bounding bounding box for floating point coordinates and integer indices
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <cassert>
#include <cmath>

#include "cstone/cuda/annotation.hpp"
#include "cstone/primitives/stl.hpp"
#include "cstone/tree/definitions.h"

#include "common.hpp"

namespace cstone
{

/*! @brief normalize a spatial length w.r.t. to a min/max range
 *
 * @tparam T
 * @param d
 * @param min
 * @param max
 * @return
 */
template<class T>
HOST_DEVICE_FUN constexpr T normalize(T d, T min, T max) { return (d - min) / (max - min); }

/*! @brief map x into periodic range 0...R-1
 *
 * @tparam R periodic range
 * @param x  input value
 * @return   x mapped into periodic range
 *
 * Examples:
 *   -1 -> R-1
 *    0 -> 0
 *    1 -> 1
 *  R-1 -> R-1
 *    R -> 0
 *  R+1 -> 1
 */
template<int R>
HOST_DEVICE_FUN constexpr int pbcAdjust(int x)
{
    // this version handles x outside -R, 2R
    //return x - R * std::floor(double(x)/R);
    assert(x >= -R);
    assert(x < 2 * R);
    int ret = (x < 0) ? x + R : x;
    return (ret >= R) ? ret - R : ret;
}

//! @brief maps x into the range [-R/2+1: R/2+1] (-511 to 512 with R = 1024)
template<int R>
HOST_DEVICE_FUN constexpr int pbcDistance(int x)
{
    // this version handles x outside -R, R
    //int roundAwayFromZero = (x > 0) ? x + R/2 : x - R/2;
    //return x -= R * (roundAwayFromZero / R);
    assert(x >= -R);
    assert(x <= R);
    int ret = (x <= -R/2) ? x + R : x;
    return (ret > R/2) ? ret - R : ret;
}

/*! @brief stores the coordinate bounds
 *
 * Needs a slightly different behavior in the PBC case than the existing BBox
 * to manage morton code based octrees.
 *
 * @tparam T floating point type
 */
template<class T>
class Box
{
public:
    HOST_DEVICE_FUN constexpr
    Box(T xyzMin, T xyzMax, bool hasPbc = false) :
        limits{xyzMin, xyzMax, xyzMin, xyzMax, xyzMin, xyzMax},
        lengths_{xyzMax-xyzMin, xyzMax-xyzMin, xyzMax-xyzMin},
        inverseLengths_{T(1.)/(xyzMax-xyzMin), T(1.)/(xyzMax-xyzMin), T(1.)/(xyzMax-xyzMin)},
        pbc{hasPbc, hasPbc, hasPbc}
    {}

    HOST_DEVICE_FUN constexpr
    Box(T xmin, T xmax, T ymin, T ymax, T zmin, T zmax,
        bool pbcX = false, bool pbcY = false, bool pbcZ = false)
    : limits{xmin, xmax, ymin, ymax, zmin, zmax},
      lengths_{xmax-xmin, ymax-ymin, zmax-zmin},
      inverseLengths_{T(1.)/(xmax-xmin), T(1.)/(ymax-ymin), T(1.)/(zmax-zmin)},
      pbc{pbcX, pbcY, pbcZ}
    {}

    HOST_DEVICE_FUN constexpr T xmin() const { return limits[0]; }
    HOST_DEVICE_FUN constexpr T xmax() const { return limits[1]; }
    HOST_DEVICE_FUN constexpr T ymin() const { return limits[2]; }
    HOST_DEVICE_FUN constexpr T ymax() const { return limits[3]; }
    HOST_DEVICE_FUN constexpr T zmin() const { return limits[4]; }
    HOST_DEVICE_FUN constexpr T zmax() const { return limits[5]; }

    //! @brief return edge lengths
    HOST_DEVICE_FUN constexpr T lx() const { return lengths_[0]; }
    HOST_DEVICE_FUN constexpr T ly() const { return lengths_[1]; }
    HOST_DEVICE_FUN constexpr T lz() const { return lengths_[2]; }

    //! @brief return inverse edge lengths
    HOST_DEVICE_FUN constexpr T ilx() const { return inverseLengths_[0]; }
    HOST_DEVICE_FUN constexpr T ily() const { return inverseLengths_[1]; }
    HOST_DEVICE_FUN constexpr T ilz() const { return inverseLengths_[2]; }

    HOST_DEVICE_FUN constexpr bool pbcX() const { return pbc[0]; } // NOLINT
    HOST_DEVICE_FUN constexpr bool pbcY() const { return pbc[1]; } // NOLINT
    HOST_DEVICE_FUN constexpr bool pbcZ() const { return pbc[2]; } // NOLINT

    //! @brief return the shortest coordinate range in any dimension
    HOST_DEVICE_FUN constexpr T minExtent() const
    {
        return stl::min(stl::min(lengths_[0], lengths_[1]), lengths_[2]);
    }

    //! @brief return the longes coordinate range in any dimension
    HOST_DEVICE_FUN constexpr T maxExtent() const
    {
        return stl::max(stl::max(lengths_[0], lengths_[1]), lengths_[2]);
    }

private:
    HOST_DEVICE_FUN
    friend constexpr bool operator==(const Box<T>& a, const Box<T>& b)
    {
        return    a.limits[0] == b.limits[0]
               && a.limits[1] == b.limits[1]
               && a.limits[2] == b.limits[2]
               && a.limits[3] == b.limits[3]
               && a.limits[4] == b.limits[4]
               && a.limits[5] == b.limits[5]
               && a.pbc[0] == b.pbc[0]
               && a.pbc[1] == b.pbc[1]
               && a.pbc[2] == b.pbc[2];
    }

    T limits[6];
    T lengths_[3];
    T inverseLengths_[3];
    bool pbc[3];
};

//! @brief Fold X into periodic boundaries,
template<class T>
HOST_DEVICE_FUN inline Vec3<T> applyPbc(Vec3<T> X, const Box<T>& box)
{
    X[0] -= box.pbcX() * box.lx() * std::rint(X[0] * box.ilx());
    X[1] -= box.pbcY() * box.ly() * std::rint(X[1] * box.ily());
    X[2] -= box.pbcZ() * box.lz() * std::rint(X[2] * box.ilz());

    return X;
}

//! @brief Legacy PBC
template<class T>
HOST_DEVICE_FUN inline void applyPBC(const cstone::Box<T>& box, T r, T& xx, T& yy, T& zz)
{
    if (box.pbcX() && xx > r)
        xx -= box.lx();
    else if (box.pbcX() && xx < -r)
        xx += box.lx();

    if (box.pbcY() && yy > r)
        yy -= box.ly();
    else if (box.pbcY() && yy < -r)
        yy += box.ly();

    if (box.pbcZ() && zz > r)
        zz -= box.lz();
    else if (box.pbcZ() && zz < -r)
        zz += box.lz();

    // xx += bbox.PBCx * ((xx < -r) - (xx > r)) * (bbox.xmax-bbox.xmin);
    // yy += bbox.PBCy * ((yy < -r) - (yy > r)) * (bbox.ymax-bbox.ymin);
    // zz += bbox.PBCz * ((zz < -r) - (zz > r)) * (bbox.zmax-bbox.zmin);
}

template<class T>
HOST_DEVICE_FUN inline T distancePBC(const cstone::Box<T>& box, T hi, T x1, T y1, T z1, T x2, T y2, T z2)
{
    T xx = x1 - x2;
    T yy = y1 - y2;
    T zz = z1 - z2;

    applyPBC<T>(box, 2.0 * hi, xx, yy, zz);

    return std::sqrt(xx * xx + yy * yy + zz * zz);
}

/*! @brief stores octree index integer bounds
 */
template<class T>
class SimpleBox
{
public:
    HOST_DEVICE_FUN constexpr
    SimpleBox() : limits{0,0,0,0,0,0} {}

    HOST_DEVICE_FUN constexpr
    SimpleBox(T xyzMin, T xyzMax) :
        limits{xyzMin, xyzMax, xyzMin, xyzMax, xyzMin, xyzMax}
    {}

    HOST_DEVICE_FUN constexpr
    SimpleBox(T xmin, T xmax, T ymin, T ymax, T zmin, T zmax)
        : limits{xmin, xmax, ymin, ymax, zmin, zmax}
    {}

    HOST_DEVICE_FUN constexpr T xmin() const { return limits[0]; } // NOLINT
    HOST_DEVICE_FUN constexpr T xmax() const { return limits[1]; } // NOLINT
    HOST_DEVICE_FUN constexpr T ymin() const { return limits[2]; } // NOLINT
    HOST_DEVICE_FUN constexpr T ymax() const { return limits[3]; } // NOLINT
    HOST_DEVICE_FUN constexpr T zmin() const { return limits[4]; } // NOLINT
    HOST_DEVICE_FUN constexpr T zmax() const { return limits[5]; } // NOLINT

    //! @brief return the shortest coordinate range in any dimension
    HOST_DEVICE_FUN constexpr T minExtent() const // NOLINT
    {
        return stl::min(stl::min(xmax() - xmin(), ymax() - ymin()), zmax() - zmin());
    }

private:
    HOST_DEVICE_FUN
    friend constexpr bool operator==(const SimpleBox& a, const SimpleBox& b)
    {
        return    a.limits[0] == b.limits[0]
                  && a.limits[1] == b.limits[1]
                  && a.limits[2] == b.limits[2]
                  && a.limits[3] == b.limits[3]
                  && a.limits[4] == b.limits[4]
                  && a.limits[5] == b.limits[5];
    }

    HOST_DEVICE_FUN
    friend bool operator<(const SimpleBox& a, const SimpleBox& b)
    {
        return util::tie(a.limits[0], a.limits[1], a.limits[2], a.limits[3], a.limits[4], a.limits[5]) <
               util::tie(b.limits[0], b.limits[1], b.limits[2], b.limits[3], b.limits[4], b.limits[5]);
    }

    T limits[6];
};

using IBox = SimpleBox<int>;

template<class T>
using FBox = SimpleBox<T>;

template<class KeyType, class T>
constexpr HOST_DEVICE_FUN util::tuple<Vec3<T>, Vec3<T>> centerAndSize(const IBox& ibox, const Box<T>& box)
{
    constexpr int maxCoord = 1u << maxTreeLevel<KeyType>{};
    // smallest octree cell edge length in unit cube
    constexpr T uL = T(1.) / maxCoord;

    T halfUnitLengthX = T(0.5) * uL * box.lx();
    T halfUnitLengthY = T(0.5) * uL * box.ly();
    T halfUnitLengthZ = T(0.5) * uL * box.lz();
    Vec3<T> boxCenter = {box.xmin() + (ibox.xmax() + ibox.xmin()) * halfUnitLengthX,
                         box.ymin() + (ibox.ymax() + ibox.ymin()) * halfUnitLengthY,
                         box.zmin() + (ibox.zmax() + ibox.zmin()) * halfUnitLengthZ};
    Vec3<T> boxSize = {(ibox.xmax() - ibox.xmin()) * halfUnitLengthX,
                       (ibox.ymax() - ibox.ymin()) * halfUnitLengthY,
                       (ibox.zmax() - ibox.zmin()) * halfUnitLengthZ};

    return { boxCenter, boxSize };
}

/*! @brief create a floating point box from and integer box
 *
 * @tparam T         float or double
 * @tparam KeyType   32- or 64-bit unsigned integer
 * @param ibox       integer box
 * @param box        global coordinate bounding box
 * @return           the floating point box
 */
template<class KeyType, class T>
constexpr HOST_DEVICE_FUN FBox<T> createFpBox(const IBox& ibox, const Box<T>& box)
{
    constexpr int maxCoord = 1u << maxTreeLevel<KeyType>{};
    // smallest octree cell edge length in unit cube
    constexpr T uL = T(1.) / maxCoord;

    T unitLengthX = uL * box.lx();
    T unitLengthY = uL * box.ly();
    T unitLengthZ = uL * box.lz();
    T xMin = box.xmin() + ibox.xmin() * unitLengthX;
    T yMin = box.ymin() + ibox.ymin() * unitLengthY;
    T zMin = box.zmin() + ibox.zmin() * unitLengthZ;
    T xMax = box.xmin() + ibox.xmax() * unitLengthX;
    T yMax = box.ymin() + ibox.ymax() * unitLengthY;
    T zMax = box.zmin() + ibox.zmax() * unitLengthZ;

    return {xMin, xMax, yMin, yMax, zMin, zMax};
}

} // namespace cstone
