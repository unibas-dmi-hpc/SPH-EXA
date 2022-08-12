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
 * @brief  Build a tree for Ryoanji with the cornerstone framework
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <memory>

#include "ryoanji/nbody/types.h"

namespace ryoanji
{

template<class KeyType>
class TreeBuilder
{
public:
    TreeBuilder();

    ~TreeBuilder();

    //! @brief initialize with the desired maximum particles per leaf cell
    TreeBuilder(unsigned ncrit);

    /*! @brief construct an octree from body coordinates
     *
     * @tparam        T           float or double
     * @param[inout]  x           body x-coordinates, will be sorted in SFC-order
     * @param[inout]  y           body y-coordinates, will be sorted in SFC-order
     * @param[inout]  z           body z-coordinates, will be sorted in SFC-order
     * @param[in]     numBodies   number of bodies in @p x,y,z
     * @param[in]     box         the coordinate bounding box
     * @return                    the total number of cells in the constructed octree
     *
     * Note: x,y,z arrays will be sorted in SFC order to match be consistent with the cell body offsets of the tree
     */
    template<class T>
    int update(T* x, T* y, T* z, size_t numBodies, const cstone::Box<T>& box);

    /*! @brief extract the octree level range from the previous update call
     *
     * @param[out] h_levelRange  indices of the first node at each subdivison level
     * @return     the maximum subdivision level in the output tree
     */
    int extract(int2* h_levelRange);

    const LocalIndex*    layout() const;
    const TreeNodeIndex* childOffsets() const;
    const TreeNodeIndex* leafToInternal() const;
    const TreeNodeIndex* internalToLeaf() const;

    TreeNodeIndex numLeafNodes() const;
    unsigned      maxTreeLevel() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

extern template class TreeBuilder<uint32_t>;
extern template class TreeBuilder<uint64_t>;

} // namespace ryoanji
