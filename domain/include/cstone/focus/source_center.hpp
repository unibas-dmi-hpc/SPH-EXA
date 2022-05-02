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
 * @brief Compute leaf cell source centers based on local information
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/util/array.hpp"
#include "cstone/tree/octree_internal.hpp"

namespace cstone
{

template<class T>
using SourceCenterType = util::array<T, 4>;

//! @brief add a single body contribution to a mass center
template<class T>
HOST_DEVICE_FUN void addBody(SourceCenterType<T>& center, const SourceCenterType<T>& source)
{
    T weight = source[3];

    center[0] += weight * source[0];
    center[1] += weight * source[1];
    center[2] += weight * source[2];
    center[3] += weight;
}

//! @brief finish mass center computation by diving coordinates by total mass
template<class T>
HOST_DEVICE_FUN SourceCenterType<T> normalizeMass(SourceCenterType<T> center)
{
    T invM = (center[3] != T(0.0)) ? T(1.0) / center[3] : T(0.0);
    center[0] *= invM;
    center[1] *= invM;
    center[2] *= invM;

    return center;
}

//! @brief compute a mass center from particles
template<class Ts, class Tc, class Tm>
HOST_DEVICE_FUN SourceCenterType<Ts>
massCenter(const Tc* x, const Tc* y, const Tc* z, const Tm* m, LocalIndex first, LocalIndex last)
{
    SourceCenterType<Ts> center{0, 0, 0, 0};
    for (LocalIndex i = first; i < last; ++i)
    {
        addBody(center, SourceCenterType<Ts>{Ts(x[i]), Ts(y[i]), Ts(z[i]), Ts(m[i])});
    }

    return normalizeMass(center);
}

//! @brief compute a mass center from other mass centers for use in tree upsweep
template<class T>
struct CombineSourceCenter
{
    SourceCenterType<T> operator()(TreeNodeIndex /*nodeIdx*/, TreeNodeIndex child, const SourceCenterType<T>* centers)
    {
        SourceCenterType<T> center{0, 0, 0, 0};

        for (TreeNodeIndex i = child; i < child + 8; ++i)
        {
            addBody(center, centers[i]);
        }
        return normalizeMass(center);
    }
};

template<class T1, class T2, class T3, class KeyType>
void computeLeafMassCenter(gsl::span<const T1> x,
                           gsl::span<const T1> y,
                           gsl::span<const T1> z,
                           gsl::span<const T2> m,
                           gsl::span<const KeyType> particleKeys,
                           const Octree<KeyType>& octree,
                           gsl::span<SourceCenterType<T3>> sourceCenter)
{
#pragma omp parallel for
    for (TreeNodeIndex nodeIdx = 0; nodeIdx < octree.numTreeNodes(); ++nodeIdx)
    {
        if (octree.isLeaf(nodeIdx))
        {
            KeyType nodeStart = octree.codeStart(nodeIdx);
            KeyType nodeEnd   = octree.codeEnd(nodeIdx);

            // find elements belonging to particles in node i
            LocalIndex first = findNodeAbove(particleKeys, nodeStart);
            LocalIndex last  = findNodeAbove(particleKeys, nodeEnd);

            sourceCenter[nodeIdx] = massCenter<T3>(x.data(), y.data(), z.data(), m.data(), first, last);
        }
    }
}

template<class T, class KeyType>
HOST_DEVICE_FUN T computeMac(KeyType prefix, Vec3<T> expCenter, float invTheta, const Box<T>& box)
{
    KeyType nodeKey  = decodePlaceholderBit(prefix);
    int prefixLength = decodePrefixLength(prefix);

    IBox cellBox              = sfcIBox(sfcKey(nodeKey), prefixLength / 3);
    auto [geoCenter, geoSize] = centerAndSize<KeyType>(cellBox, box);

    Vec3<T> dX = expCenter - geoCenter;

    T s   = sqrt(norm2(dX));
    T l   = T(2.0) * max(geoSize);
    T mac = l * invTheta + s;

    return mac * mac;
}

//! @brief replace the last center element (mass) with the squared mac radius
template<class T, class KeyType>
void setMac(gsl::span<const KeyType> nodeKeys,
            gsl::span<SourceCenterType<T>> centers,
            float invTheta,
            const Box<T>& box)
{
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < nodeKeys.size(); ++i)
    {
        Vec4<T> center = centers[i];
        T mac          = computeMac(nodeKeys[i], util::makeVec3(center), invTheta, box);
        centers[i][3]  = (center[3] != T(0)) ? mac : T(0);
    }
}

} // namespace cstone
