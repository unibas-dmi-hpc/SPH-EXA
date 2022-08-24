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

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/primitives/primitives_gpu.h"
#include "cstone/util/gsl-lite.hpp"
#include "cstone/util/reallocate.hpp"
#include "domaindecomp.hpp"

namespace cstone
{

/*! @brief Based on global assignment, create the list of local particle index ranges to send to each rank
 *
 * @tparam    KeyType        32- or 64-bit integer
 * @param[in] assignment     global space curve assignment to ranks
 * @param[in] tree           global cornerstone octree that matches the node counts used to create @p assignment
 * @param[in] particleKeys   sorted list of SFC keys of local particles present on this rank, ON DEVICE
 * @param[-]  sendScratch    array of length assignment.numRanks() to store search keys, uninitialized, ON DEVICE
 * @param[-]  receiveScratch array of length assignment.numRanks() to store search results, uninitialized, ON DEVICE
 * @return                   for each rank, a list of index ranges into @p particleKeys to send
 *
 * Converts the global assignment particle keys ranges into particle indices with binary search
 */
template<class KeyType, class DeviceVector>
SendList createSendListGpu(const SpaceCurveAssignment& assignment,
                           gsl::span<const KeyType> treeLeaves,
                           gsl::span<const KeyType> particleKeys,
                           DeviceVector& sendScratch,
                           DeviceVector& receiveScratch)
{
    size_t numRanks = assignment.numRanks();
    using IndexType = SendManifest::IndexType;

    size_t ssz = reallocateDeviceBytes(sendScratch, numRanks * sizeof(KeyType));
    size_t rsz = reallocateDeviceBytes(receiveScratch, numRanks * sizeof(LocalIndex));
    gsl::span<KeyType> d_searchKeys{reinterpret_cast<KeyType*>(rawPtr(sendScratch)), numRanks};
    gsl::span<LocalIndex> d_indices{reinterpret_cast<LocalIndex*>(rawPtr(receiveScratch)), numRanks};

    SendList ret(numRanks);

    std::vector<KeyType> searchKeys(numRanks);
    for (int rank = 0; rank < numRanks; ++rank)
    {
        searchKeys[rank] = treeLeaves[assignment.firstNodeIdx(rank)];
    }

    memcpyH2D(searchKeys.data(), searchKeys.size(), d_searchKeys.data());
    lowerBoundGpu(particleKeys.begin(), particleKeys.end(), d_searchKeys.begin(), d_searchKeys.end(),
                  d_indices.begin());

    std::vector<IndexType> indices(numRanks + 1, particleKeys.size());
    memcpyD2H(d_indices.data(), numRanks, indices.data());

    for (int rank = 0; rank < numRanks; ++rank)
    {
        ret[rank].addRange(indices[rank], indices[rank + 1]);
    }

    reallocateDevice(sendScratch, ssz, 1.0);
    reallocateDevice(receiveScratch, rsz, 1.0);

    return ret;
}

} // namespace cstone
