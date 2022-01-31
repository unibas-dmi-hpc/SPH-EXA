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

template<class Ts, class Tc, class Tm>
util::array<Ts, 4> massCenter(gsl::span<const Tc> x,
                              gsl::span<const Tc> y,
                              gsl::span<const Tc> z,
                              gsl::span<const Tm> m,
                              LocalIndex first,
                              LocalIndex last)
{
    util::array<Ts, 4> center{0, 0, 0, 0};
    for (LocalIndex i = first; i < last; ++i)
    {
        Tm weight = m[i];

        center[0] += weight * x[i];
        center[1] += weight * y[i];
        center[2] += weight * z[i];
        center[3] += weight;
    }
    Tm invM = (center[3] != Tm(0.0)) ? Tm(1.0) / center[3] : Tm(0.0);
    center[0] *= invM;
    center[1] *= invM;
    center[2] *= invM;

    return center;
}

template<class T>
util::array<T, 4> massCenter(const util::array<T, 4>* firstSource, const util::array<T, 4>* lastSource)
{
    util::array<T, 4> center{0, 0, 0, 0};
    for ( ; firstSource != lastSource; ++firstSource)
    {
        auto source = *firstSource;
        T weight = source[3];

        center[0] += weight * source[0];
        center[1] += weight * source[1];
        center[2] += weight * source[2];
        center[3] += weight;
    }

    T invM = (center[3] != T(0.0)) ? T(1.0) / center[3] : T(0.0);
    center[0] *= invM;
    center[1] *= invM;
    center[2] *= invM;

    return center;
}

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

template<class T, class KeyType>
void upsweepMassCenter(const Octree<KeyType>& octree, gsl::span<util::array<T, 4>> centers)
{
    int currentLevel = maxTreeLevel<KeyType>{};

    for ( ; currentLevel >= 0; --currentLevel)
    {
        TreeNodeIndex start = octree.levelOffset(currentLevel);
        TreeNodeIndex end   = octree.levelOffset(currentLevel + 1);
        #pragma omp parallel for schedule(static)
        for (TreeNodeIndex i = start; i < end; ++i)
        {
            if (!octree.isLeaf(i))
            {
                auto* first = centers.data() + octree.child(i, 0);
                centers[i] = massCenter(first, first + 8);
            }
        }
    }
}

} // namespace cstone
