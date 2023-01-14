/*
 * MIT License
 *
 * Copyright (c) 2022 CSCS, ETH Zurich
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
 * @brief  Compute cell mass centers for use in focus tree
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/tree/definitions.h"

namespace cstone
{

/*! @brief compute mass centers of leaf cells
 *
 * @param[in]  x                particle x coordinates
 * @param[in]  y                particle y coordinates
 * @param[in]  z                particle z coordinates
 * @param[in]  m                particle masses
 * @param[in]  leafToInternal   maps a leaf node index to an internal layout node index
 * @param[in]  numLeaves        number of leaf nodes
 * @param[in]  layout           particle location of each node, length @a numLeaves + 1
 * @param[out] centers          output mass centers, in internal node layout, length >= max(leafToInternal)
 */
template<class Tc, class Tm, class Tf>
extern void computeLeafSourceCenterGpu(const Tc* x,
                                       const Tc* y,
                                       const Tc* z,
                                       const Tm* m,
                                       const TreeNodeIndex* leafToInternal,
                                       TreeNodeIndex numLeaves,
                                       const LocalIndex* layout,
                                       Vec4<Tf>* centers);

/*! @brief compute center of gravity for internal nodes with an upsweep
 *
 * @tparam T                   float or double
 * @param[in]    numLevels     max tree depth
 * @param[in]    levelRange    first node index per tree level
 * @param[in]    childOffsets  indices of first child node of each node
 * @param[inout] centers       center of mass coordinates with leaf node centers set
 */
template<class T>
extern void upsweepCentersGpu(int numLevels,
                              const TreeNodeIndex* levelRange,
                              const TreeNodeIndex* childOffsets,
                              SourceCenterType<T>* centers);

//! @brief compute geometric node center and sizes based on node SFC keys
template<class KeyType, class T>
extern void computeGeoCentersGpu(
    const KeyType* prefixes, TreeNodeIndex numNodes, Vec3<T>* centers, Vec3<T>* sizes, const Box<T>& box);

} // namespace cstone