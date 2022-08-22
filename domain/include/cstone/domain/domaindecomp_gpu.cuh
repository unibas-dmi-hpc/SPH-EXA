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
 * @brief Functions to assign a global cornerstone octree to different ranks
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * Any code in this file relies on a global cornerstone octree on each calling rank.
 */

#pragma once

#include <thrust/device_ptr.h>

#include "cstone/primitives/primitives_gpu.hpp"
#include "cstone/util/gsl-lite.hpp"
#include "domaindecomp.hpp"

namespace cstone
{

/*! @brief Based on global assignment, create the list of local particle index ranges to send to each rank
 *
 * @tparam KeyType      32- or 64-bit integer
 * @param[in] assignment    global space curve assignment to ranks
 * @param[in] tree          global cornerstone octree that matches the node counts used to create @p assignment
 * @param[in] particleKeys  sorted list of SFC keys of local particles present on this rank, ON DEVICE
 * @param[-]  d_searchKeys  array of length assignment.numRanks() to store search keys, uninitialized, ON DEVICE
 * @param[-]  d_indices     array of length assignment.numRanks() to store search results, uninitialized, ON DEVICE
 * @return                  for each rank, a list of index ranges into @p particleKeys to send
 *
 * Converts the global assignment particle keys ranges into particle indices with binary search
 */
template<class KeyType>
SendList createSendListGpu(const SpaceCurveAssignment& assignment,
                           gsl::span<const KeyType> treeLeaves,
                           gsl::span<const KeyType> particleKeys,
                           gsl::span<KeyType> d_searchKeys,
                           gsl::span<LocalIndex> d_indices)
{
    int numRanks    = assignment.numRanks();
    using IndexType = SendManifest::IndexType;

    assert(d_searchKeys.size() == numRanks);
    assert(d_indices.size() == numRanks);

    SendList ret(numRanks);

    std::vector<KeyType> searchKeys(numRanks);
    for (int rank = 0; rank < numRanks; ++rank)
    {
        searchKeys[rank] = treeLeaves[assignment.firstNodeIdx(rank)];
    }

    thrust::copy(searchKeys.begin(), searchKeys.end(), thrust::device_pointer_cast(d_searchKeys.data()));
    lowerBoundGpu(particleKeys.begin(), particleKeys.end(), d_searchKeys.begin(), d_searchKeys.end(),
                  d_indices.begin());

    std::vector<IndexType> indices(numRanks + 1, particleKeys.size());
    thrust::copy_n(thrust::device_pointer_cast(d_indices.data()), numRanks, indices.begin());

    for (int rank = 0; rank < numRanks; ++rank)
    {
        ret[rank].addRange(indices[rank], indices[rank + 1]);
    }

    return ret;
}

} // namespace cstone
