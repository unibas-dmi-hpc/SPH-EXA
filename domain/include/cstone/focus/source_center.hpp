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

template<class T1, class T2, class T3, class KeyType>
void computeLocalSourceCenter(const T1* x,
                              const T1* y,
                              const T1* z,
                              const T2* m,
                              const KeyType* particleKeys,
                              LocalIndex numParticles,
                              const Octree<KeyType>& octree,
                              util::array<T3, 4>* sourceCenter)
{
    #pragma omp parallel for
    for (TreeNodeIndex nodeIdx = 0; nodeIdx < octree.numTreeNodes(); ++nodeIdx)
    {
        if (octree.isLeaf(nodeIdx))
        {
            KeyType nodeStart = octree.codeStart(nodeIdx);
            KeyType nodeEnd   = octree.codeEnd(nodeIdx);

            // find elements belonging to particles in node i
            LocalIndex startIndex = findNodeAbove(particleKeys, nodeStart);
            LocalIndex endIndex   = findNodeAbove(particleKeys, nodeEnd);

            util::array<T3, 4> center{0, 0, 0, 0};
            for (LocalIndex i = startIndex; i < endIndex; ++i)
            {
                T3 weight = m[i];

                center[0] += weight * x[i];
                center[1] += weight * y[i];
                center[2] += weight * z[i];
                center[3] += weight;
            }
            T3 invM = (center[3] != T3(0.0)) ? T3(1.0) / center[3] : T3(0.0);
            center[0] *= invM;
            center[1] *= invM;
            center[2] *= invM;

            sourceCenter[nodeIdx] = center;
        }
    }
}

} // namespace cstone
