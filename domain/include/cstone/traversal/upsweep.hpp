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
 * @brief  Generic octree upsweep procedure to calculate quantities for internal nodes from their children
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * Examples applications would be the calculation of particle counts for internal nodes for given leaf counts,
 * the maximum smoothing length of any particle in the node or multipole moments.
 */

#pragma once

#include "cstone/tree/octree_internal.hpp"

namespace cstone
{

/*! @brief performs an upsweep operation, calculates quantities for internal nodes, based on given leaf nodes
 *
 * @tparam     T                     anything that can be copied
 * @tparam     KeyType               32- or 64-bit unsigned integer
 * @tparam     CombinationFunction   callable with signature T(T,T,T,T,T,T,T,T)
 * @param[in]  octree                the octree
 * @param[in]  leafQuantities        input array of length octree.numLeafNodes()
 * @param[out] internalQuantities    output array of length octree.numInternalNodes()
 * @param[in]  combinationFunction   callable of type @p CombinationFunction
 */
template<class T, class KeyType, class CombinationFunction>
void upsweep(const Octree<KeyType>& octree, const T* leafQuantities, T* internalQuantities, CombinationFunction combinationFunction)
{
    int depth = 1;
    TreeNodeIndex internalNodeIndex = octree.numInternalNodes();

    internalNodeIndex -= octree.numTreeNodes(depth);
    #pragma omp parallel for schedule(static)
    for (TreeNodeIndex i = internalNodeIndex; i < internalNodeIndex + octree.numTreeNodes(depth); ++i)
    {
        internalQuantities[i] = combinationFunction(leafQuantities[octree.childDirect(i, 0)],
                                                    leafQuantities[octree.childDirect(i, 1)],
                                                    leafQuantities[octree.childDirect(i, 2)],
                                                    leafQuantities[octree.childDirect(i, 3)],
                                                    leafQuantities[octree.childDirect(i, 4)],
                                                    leafQuantities[octree.childDirect(i, 5)],
                                                    leafQuantities[octree.childDirect(i, 6)],
                                                    leafQuantities[octree.childDirect(i, 7)]);
    }

    depth++;

    while (depth < maxTreeLevel<KeyType>{} && octree.numTreeNodes(depth) > 0)
    {
        internalNodeIndex -= octree.numTreeNodes(depth);
        #pragma omp parallel for schedule(static)
        for (TreeNodeIndex i = internalNodeIndex; i < internalNodeIndex + octree.numTreeNodes(depth); ++i)
        {
            const T& a = octree.isLeafChild(i, 0) ? leafQuantities[octree.childDirect(i, 0)]
                                                  : internalQuantities[octree.childDirect(i, 0)];
            const T& b = octree.isLeafChild(i, 1) ? leafQuantities[octree.childDirect(i, 1)]
                                                  : internalQuantities[octree.childDirect(i, 1)];
            const T& c = octree.isLeafChild(i, 2) ? leafQuantities[octree.childDirect(i, 2)]
                                                  : internalQuantities[octree.childDirect(i, 2)];
            const T& d = octree.isLeafChild(i, 3) ? leafQuantities[octree.childDirect(i, 3)]
                                                  : internalQuantities[octree.childDirect(i, 3)];
            const T& e = octree.isLeafChild(i, 4) ? leafQuantities[octree.childDirect(i, 4)]
                                                  : internalQuantities[octree.childDirect(i, 4)];
            const T& f = octree.isLeafChild(i, 5) ? leafQuantities[octree.childDirect(i, 5)]
                                                  : internalQuantities[octree.childDirect(i, 5)];
            const T& g = octree.isLeafChild(i, 6) ? leafQuantities[octree.childDirect(i, 6)]
                                                  : internalQuantities[octree.childDirect(i, 6)];
            const T& h = octree.isLeafChild(i, 7) ? leafQuantities[octree.childDirect(i, 7)]
                                                  : internalQuantities[octree.childDirect(i, 7)];

            internalQuantities[i] = combinationFunction(a, b, c, d, e, f, g, h);
        }

        depth++;
    }
}

/*! @brief convenience wrapper for contiguous property arrays
 *
 * @param property  array of length octree.numTreeNodes(), with elements corresponding to leaf cells
 *                  (in the range [octree.numInternalNodes():octree.numTreeNodes()]) already computed.
 *
 * Upon return, internal cells (in the range [0:octree.numInternalNodes()]) of @p property are computed
 * with an upsweep using the supplied combination function.
 */
template<class T, class KeyType, class CombinationFunction>
inline void upsweep(const Octree<KeyType>& octree, T* property, CombinationFunction combinationFunction)
{
    upsweep(octree, property + octree.numInternalNodes(), property, combinationFunction);
}

} // namespace cstone
