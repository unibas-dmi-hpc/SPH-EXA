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

/*! \file
 * \brief  Generic octree upsweep procedure to calculate quantities for internal nodes from their children
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * Examples would be the calculation of particle counts for internal nodes for given leaf counts or
 * or the maximum smoothing length of any particle in the node.
 */

#pragma once

#include "octree_internal.hpp"

namespace cstone
{

template<class T>
using CombinationFunction = T (*)(const T*, const T*, const T*, const T*, const T*, const T*, const T*, const T*);


template<class T, class I>
void upsweep(const Octree<I>& octree, T* internalQuantities, const T* leafQuantities, CombinationFunction<T> combinationFunction)
{
    TreeNodeIndex nLeaves = octree.nLeaves();

    for (TreeNodeIndex i1 = 0; i1 < nLeaves/8; ++i1)
    {
        TreeNodeIndex nodeIdx = octree.parent(i1 * 8);
    }
}

} // namespace cstone