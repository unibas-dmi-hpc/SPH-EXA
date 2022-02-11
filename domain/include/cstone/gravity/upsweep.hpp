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

#include "cstone/focus/source_center.hpp"
#include "cstone/gravity/multipole.hpp"
#include "cstone/tree/octree_internal.hpp"

namespace cstone
{

/*! @brief compute multipoles from particle data for the entire tree hierarchy
 *
 * @tparam     KeyType    32- or 64-bit unsigned integer
 * @tparam     T1         float or double
 * @tparam     T2         float or double
 * @tparam     MType      Multipole type, e.g. CartesianQuadrupole
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
template<class KeyType, class T1, class T2, class MType>
void computeLeafMultipoles(const Octree<KeyType>& octree,
                           gsl::span<const LocalIndex> layout,
                           const T1* x,
                           const T1* y,
                           const T1* z,
                           const T2* m,
                           const SourceCenterType<T1>* centers,
                           MType* multipoles)
{
    #pragma omp parallel for schedule(static)
    for (TreeNodeIndex leafIdx = 0; leafIdx < octree.numLeafNodes(); ++leafIdx)
    {
        TreeNodeIndex i = octree.toInternal(leafIdx);
        particle2Multipole(x, y, z, m, layout[leafIdx], layout[leafIdx + 1], makeVec3(centers[i]), multipoles[i]);
    }
}

template<class MType>
class CombineMultipole
{
public:
    CombineMultipole(const SourceCenterType<typename MType::value_type>* centers)
        : centers_(centers)
    {
    }

    MType operator()(TreeNodeIndex nodeIdx, TreeNodeIndex child, MType* multipoles)
    {
        MType ret;
        setZero(ret);
        multipole2multipole(child, child + 8, centers_[nodeIdx], centers_, multipoles, ret);
        return ret;
    }

private:
    const SourceCenterType<typename MType::value_type>* centers_;
};

} // namespace cstone
