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
void addBody(SourceCenterType<T>& center, const SourceCenterType<T>& source)
{
    T weight = source[3];

    center[0] += weight * source[0];
    center[1] += weight * source[1];
    center[2] += weight * source[2];
    center[3] += weight;
}

//! @brief finish mass center computation by diving coordinates by total mass
template<class T>
SourceCenterType<T> normalizeMass(SourceCenterType<T> center)
{
    T invM = (center[3] != T(0.0)) ? T(1.0) / center[3] : T(0.0);
    center[0] *= invM;
    center[1] *= invM;
    center[2] *= invM;

    return center;
}

//! @brief compute a mass center from particles
template<class Ts, class Tc, class Tm>
SourceCenterType<Ts> massCenter(gsl::span<const Tc> x,
                                gsl::span<const Tc> y,
                                gsl::span<const Tc> z,
                                gsl::span<const Tm> m,
                                LocalIndex first,
                                LocalIndex last)
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
    SourceCenterType<T> operator()(const SourceCenterType<T>& a,
                                   const SourceCenterType<T>& b,
                                   const SourceCenterType<T>& c,
                                   const SourceCenterType<T>& d,
                                   const SourceCenterType<T>& e,
                                   const SourceCenterType<T>& f,
                                   const SourceCenterType<T>& g,
                                   const SourceCenterType<T>& h)
    {
        SourceCenterType<T> center{0, 0, 0, 0};
        addBody(center, a);
        addBody(center, b);
        addBody(center, c);
        addBody(center, d);
        addBody(center, e);
        addBody(center, f);
        addBody(center, g);
        addBody(center, h);
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
                           gsl::span<util::array<T3, 4>> sourceCenter)
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

            sourceCenter[nodeIdx] = massCenter<T3>(x, y, z, m, first, last);
        }
    }
}

} // namespace cstone
