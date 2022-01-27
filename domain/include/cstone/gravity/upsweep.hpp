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
 * @brief construction of gravity data for a given octree and particle coordinates
 *
 * @author Sebastian Keller        <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/tree/octree_internal.hpp"
#include "cstone/gravity/multipole.hpp"

namespace cstone
{

/*! @brief compute multipoles from particle data for the entire tree hierarchy
 *
 * @tparam     KeyType    32- or 64-bit unsigned integer
 * @tparam     T1         float or double
 * @tparam     T2         float or double
 * @tparam     T3         float or double
 * @param[in]  octree     full linked octree
 * @param[in]  layout     array of length @p octree.numLeafNodes()+1, layout[i] is the start offset
 *                        into the x,y,z,m arrays for the leaf node with index i. The last element
 *                        is equal to the length of the x,y,z,m arrays.
 * @param[in]  x          local particle x-coordinates
 * @param[in]  y          local particle y-coordinates
 * @param[in]  z          local particle z-coordinates
 * @param[in]  m          local particle masses
 * @param[out] multipoles output multipole moments
 */
template<class KeyType, class T1, class T2, class T3>
void computeMultipoles(const TdOctree<KeyType>& octree, gsl::span<const LocalIndex> layout,
                       const T1* x, const T1* y, const T1* z, const T2* m, GravityMultipole<T3>* multipoles)
{
    // calculate multipoles for leaf cells
    #pragma omp parallel for schedule(static)
    for (TreeNodeIndex i = 0; i < octree.numLeafNodes(); ++i)
    {
        LocalIndex startIndex   = layout[i];
        LocalIndex numParticles = layout[i + 1] - startIndex;

        TreeNodeIndex fullIndex = octree.toInternal(i);
        multipoles[fullIndex] =
            particle2Multipole<T3>(x + startIndex, y + startIndex, z + startIndex, m + startIndex, numParticles);
    }

    // calculate internal cells from leaf cells
    upsweep(octree, multipoles, multipole2multipole<T3>);
}

} // namespace cstone
