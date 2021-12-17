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
 * @brief Binary search based determination of tree node indices to request from remote ranks
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <vector>

#include "cstone/domain/domaindecomp.hpp"
#include "cstone/tree/octree.hpp"
#include "cstone/util/gsl-lite.hpp"
#include "cstone/util/index_ranges.hpp"

namespace cstone
{

/*! @brief determine indices in the local focus tree to send out to peer ranks for counting requests
 *
 * @tparam KeyType          32- or 64-bit unsigned integer
 * @param peers             list of peer ranks (for point-to-point communication)
 * @param assignment        assignment of the global SFC to ranks
 * @param domainTreeLeaves  tree leaves of the global tree in cornerstone format
 * @param focusTreeLeaves   tree leaves of the locally focused tree in cornerstone format
 * @return                  a list of pairs (startIdx, endIdx) referring to elements of @p focusTreeLeaves
 *                          to send out to each peer rank. One of the rare cases where the range has to include
 *                          the last element. To describe n tree nodes, we need n+1 SFC keys.
 */
template<class KeyType>
std::vector<IndexPair<TreeNodeIndex>> findRequestIndices(gsl::span<const int> peers, const SpaceCurveAssignment& assignment,
                                                         gsl::span<const KeyType> domainTreeLeaves,
                                                         gsl::span<const KeyType> focusTreeLeaves)
{
    std::vector<IndexPair<TreeNodeIndex>> requestIndices;
    requestIndices.reserve(peers.size());
    for (int peer : peers)
    {
        KeyType peerSfcStart = domainTreeLeaves[assignment.firstNodeIdx(peer)];
        KeyType peerSfcEnd   = domainTreeLeaves[assignment.lastNodeIdx(peer)];

        // only fully contained nodes in [peerSfcStart:peerSfcEnd] will be requested
        TreeNodeIndex firstRequestIdx = findNodeAbove(focusTreeLeaves, peerSfcStart);
        TreeNodeIndex lastRequestIdx  = findNodeBelow(focusTreeLeaves, peerSfcEnd);

        if (lastRequestIdx < firstRequestIdx) { lastRequestIdx = firstRequestIdx; }
        requestIndices.emplace_back(firstRequestIdx, lastRequestIdx);
    }

    return requestIndices;
}

/*! @brief calculates the complementary range of the input ranges
 *
 * Input:  │      ------    -----   --     ----     --  │
 * Output: -------      ----     ---  -----    -----  ---
 *         ^                                            ^
 *         │                                            │
 * @param first                                         │
 * @param ranges   size >= 1, must be sorted            │
 * @param last    ──────────────────────────────────────/
 * @return the output ranges that cover everything within [first:last]
 *         that the input ranges did not cover
 */
std::vector<IndexPair<TreeNodeIndex>> invertRanges(TreeNodeIndex first,
                                                   gsl::span<const IndexPair<TreeNodeIndex>> ranges,
                                                   TreeNodeIndex last)
{
    assert(!ranges.empty() && std::is_sorted(ranges.begin(), ranges.end()));

    std::vector<IndexPair<TreeNodeIndex>> invertedRanges;
    if (first < ranges.front().start()) { invertedRanges.emplace_back(first, ranges.front().start()); }
    for (size_t i = 1; i < ranges.size(); ++i)
    {
        if (ranges[i-1].end() < ranges[i].start())
        {
            invertedRanges.emplace_back(ranges[i-1].end(), ranges[i].start());
        }
    }
    if (ranges.back().end() < last) { invertedRanges.emplace_back(ranges.back().end(), last); }

    return invertedRanges;
}

} // namespace cstone
