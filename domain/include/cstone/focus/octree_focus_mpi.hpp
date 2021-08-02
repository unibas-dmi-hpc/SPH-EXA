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
 * @brief Traits and functors for the MPI-enabled FocusedOctree
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/focus/exchange_focus.hpp"
#include "cstone/focus/octree_focus.hpp"

namespace cstone
{

struct MpiPeerExchange
{
    template <class KeyType>
    void operator()(gsl::span<const int> peerRanks, gsl::span<const IndexPair<TreeNodeIndex>> exchangeIndices,
                    gsl::span<const KeyType> particleKeys, gsl::span<const KeyType> localLeaves,
                    gsl::span<unsigned> localCounts)
    {
        exchangePeerCounts(peerRanks, exchangeIndices, particleKeys, localLeaves, localCounts);
    }
};

namespace focused_octree_detail
{
    struct MpiCommTag {};
}

template <class CommunicationType>
struct ExchangePeerCounts<CommunicationType, std::enable_if_t<std::is_same_v<focused_octree_detail::MpiCommTag, CommunicationType>>>
{
    using type = MpiPeerExchange;
};

template<class KeyType>
using FocusedOctree = FocusedOctreeImpl<KeyType, focused_octree_detail::MpiCommTag>;

} // namespace cstone
