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
 * @brief Functions to assign a global cornerstone octree to different ranks
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * Any code in this file relies on a global cornerstone octree on each calling rank.
 */

#pragma once

#include <algorithm>
#include <numeric>
#include <vector>

#include "cstone/tree/csarray.hpp"
#include "cstone/util/gsl-lite.hpp"
#include "index_ranges.hpp"

namespace cstone
{

/*! @brief stores which parts of the SFC belong to which rank, on a per-rank basis
 *
 * The storage layout allows fast look-up of the SFC code ranges that a given rank
 * was assigned.
 *
 * Usage constraints of this class:
 *      - Assigned ranges can be empty, but each rank has to be assigned a range (not checked)
 *      - The ranges of two consecutive ranks must not overlap and must not have holes in between,
 *        i.e. lastNodeIdx(n) == firstNodeIdx(n+1) for any rank n < nRanks-1 (checked)
 */
class SpaceCurveAssignment
{
    static constexpr TreeNodeIndex untouched = -1;

public:
    SpaceCurveAssignment()
        : rankAssignment_(1)
    {
    }

    explicit SpaceCurveAssignment(int numRanks)
        : rankAssignment_(numRanks + 1, untouched)
        , counts_(numRanks)
    {
    }

    //! @brief add an index/code range to rank @p rank
    void addRange(int rank, TreeNodeIndex lower, TreeNodeIndex upper, std::size_t cnt)
    {
        // make sure that there's no holes or overlap between or with the range of the previous rank
        assert(rankAssignment_[rank] == lower || rankAssignment_[rank] == untouched);

        rankAssignment_[rank] = lower;
        // will be overwritten by @p lower of rank+1, except if rank == numRanks-1
        rankAssignment_[rank + 1] = upper;
        counts_[rank]             = cnt;
    }

    [[nodiscard]] int numRanks() const { return int(rankAssignment_.size()) - 1; }

    [[nodiscard]] TreeNodeIndex firstNodeIdx(int rank) const { return rankAssignment_[rank]; }

    [[nodiscard]] TreeNodeIndex lastNodeIdx(int rank) const { return rankAssignment_[rank + 1]; }

    [[nodiscard]] int findRank(TreeNodeIndex nodeIdx) const
    {
        auto it = std::upper_bound(begin(rankAssignment_), end(rankAssignment_), nodeIdx);
        return int(it - begin(rankAssignment_)) - 1;
    }

    //! @brief the sum of number of particles in all ranges, i.e. total number of assigned particles per range
    [[nodiscard]] const std::size_t& totalCount(int rank) const { return counts_[rank]; }

private:
    friend bool operator==(const SpaceCurveAssignment& a, const SpaceCurveAssignment& b)
    {
        return a.rankAssignment_ == b.rankAssignment_ && a.counts_ == b.counts_;
    }

    std::vector<TreeNodeIndex> rankAssignment_;
    std::vector<size_t> counts_;
};

/*! @brief assign the global tree/SFC to nSplits ranks, assigning to each rank only a single Morton code range
 *
 * @param globalCounts       counts per leaf
 * @param nSplits            divide the global tree into nSplits pieces, sensible choice e.g.: nSplits == numRanks
 * @return                   a vector with nSplit elements, each element is a vector of SfcRanges of Morton codes
 *
 * This function acts on global data. All calling ranks should call this function with identical arguments.
 * Therefore each rank will compute the same SpaceCurveAssignment and each rank will thus know the ranges that
 * all the ranks are assigned.
 *
 */
inline SpaceCurveAssignment singleRangeSfcSplit(const std::vector<unsigned>& globalCounts, int nSplits)
{
    // one element per rank
    SpaceCurveAssignment ret(nSplits);

    std::size_t globalNParticles = std::accumulate(begin(globalCounts), end(globalCounts), std::size_t(0));

    // distribute work, every rank gets global count / nSplits,
    // the remainder gets distributed one by one
    std::vector<std::size_t> nParticlesPerSplit(nSplits, globalNParticles / nSplits);
    for (std::size_t split = 0; split < globalNParticles % nSplits; ++split)
    {
        nParticlesPerSplit[split]++;
    }

    TreeNodeIndex leavesDone = 0;
    for (int split = 0; split < nSplits; ++split)
    {
        std::size_t targetCount = nParticlesPerSplit[split];
        std::size_t splitCount  = 0;
        TreeNodeIndex j         = leavesDone;
        while (splitCount < targetCount && j < TreeNodeIndex(globalCounts.size()))
        {
            // if adding the particles of the next leaf takes us further away from
            // the target count than where we're now, we stop
            if (targetCount < splitCount + globalCounts[j] &&                          // overshoot
                targetCount - splitCount < splitCount + globalCounts[j] - targetCount) // overshoot more than undershoot
            {
                break;
            }

            splitCount += globalCounts[j++];
        }

        if (split < nSplits - 1)
        {
            // carry over difference of particles over/under assigned to next split
            // to avoid accumulating round off
            long int delta = (long int)(targetCount) - (long int)(splitCount);
            nParticlesPerSplit[split + 1] += delta;
        }
        // afaict, j < nNodes(globalTree) can only happen if there are empty nodes at the end
        else
        {
            for (; j < TreeNodeIndex(globalCounts.size()); ++j)
                splitCount += globalCounts[j];
        }

        // other distribution strategies might have more than one range per rank
        ret.addRange(split, leavesDone, j, splitCount);
        leavesDone = j;
    }

    return ret;
}

/*! @brief limit SFC range assignment transfer to the domain of the rank above or below
 *
 * @tparam        KeyType          32- or 64-bit unsigned integer
 * @param[in]     oldBoundaries    SFC key assignment boundaries to ranks from the previous step
 * @param[in]     newTree          the global octree leaves used for domain decomposition in the current step
 * @param[in]     counts           particle counts per leaf cell in @p newTree
 * @param[inout]  newAssignment    the current assignment, will be modified if needed
 *
 * When assignment boundaries change, we limit the growth of any rank downwards or upwards the SFC
 * to the previous assignment of the rank below or above, i.e. rank r can only acquire new SFC areas
 * that belonged to ranks r-1 or r+1 in the previous step. This limitation never kicks in for any
 * halfway reasonable particle configuration as the handover of a rank's entire domain to another rank
 * is quite an extreme scenario. But the limitation is useful for focused torture tests to demonstrate
 * that the domain and octree invariants still hold under such circumstances. Imposing this limitation
 * here is needed to guarantee that the focus tree resolution of any rank in its focus is not exceeded
 * in the trees of any other rank.
 */
template<class KeyType>
void limitBoundaryShifts(gsl::span<const KeyType> oldBoundaries,
                         gsl::span<const KeyType> newTree,
                         gsl::span<const unsigned> counts,
                         SpaceCurveAssignment& newAssignment)
{
    // do nothing on first call when there are no boundaries
    if (oldBoundaries.size() == 1) { return; }

    int numRanks = newAssignment.numRanks();
    std::vector<TreeNodeIndex> newIndexBoundaries(numRanks + 1, 0);
    newIndexBoundaries.back() = newAssignment.lastNodeIdx(numRanks - 1);

    bool triggerRecount = false;
    for (int rank = 1; rank < numRanks; ++rank)
    {
        KeyType newBoundary = newTree[newAssignment.firstNodeIdx(rank)];

        KeyType doNotGoBelow = oldBoundaries[rank - 1];
        KeyType doNotExceed  = oldBoundaries[rank + 1];

        TreeNodeIndex restrictedStart = newAssignment.firstNodeIdx(rank);
        if (newBoundary < doNotGoBelow)
        {
            restrictedStart = findNodeAbove(newTree.data(), newTree.size(), doNotGoBelow);
            triggerRecount  = true;
        }
        else if (newBoundary > doNotExceed)
        {
            restrictedStart = findNodeBelow(newTree.data(), newTree.size(), doNotExceed);
            triggerRecount  = true;
        }
        newIndexBoundaries[rank] = restrictedStart;
    }

    if (triggerRecount)
    {
        for (int rank = 0; rank < numRanks; ++rank)
        {
            std::size_t segmentCount = std::accumulate(counts.begin() + newIndexBoundaries[rank],
                                                       counts.begin() + newIndexBoundaries[rank + 1], std::size_t(0));
            newAssignment.addRange(rank, newIndexBoundaries[rank], newIndexBoundaries[rank + 1], segmentCount);
        }
    }
}

/*! @brief translates an assignment of a given tree to a new tree
 *
 * @tparam     KeyType         32- or 64-bit unsigned integer
 * @param[in]  assignment      domain assignment
 * @param[in]  domainTree      domain tree leaves
 * @param[in]  focusTree       focus tree leaves
 * @param[in]  peerRanks       list of peer ranks
 * @param[in]  myRank          executing rank ID
 * @param[out] focusAssignment assignment with the same SFC key ranges per
 *                             peer rank as the domain @p assignment,
 *                             but with indices valid w.r.t @p focusTree
 *
 * The focus assignment is implemented as a plain vector; since only
 * the ranges of peer ranks (and not all ranks) are set, the requirements
 * of SpaceCurveAssignment are not met and its findRank() function would not work.
 */
template<class KeyType>
void translateAssignment(const SpaceCurveAssignment& assignment,
                         gsl::span<const KeyType> domainTree,
                         gsl::span<const KeyType> focusTree,
                         gsl::span<const int> peerRanks,
                         int myRank,
                         std::vector<TreeIndexPair>& focusAssignment)
{
    focusAssignment.resize(assignment.numRanks());
    std::fill(focusAssignment.begin(), focusAssignment.end(), TreeIndexPair(0, 0));
    for (int peer : peerRanks)
    {
        KeyType peerSfcStart = domainTree[assignment.firstNodeIdx(peer)];
        KeyType peerSfcEnd   = domainTree[assignment.lastNodeIdx(peer)];

        // Note: start-end range is narrowed down if no exact match is found.
        // the discarded part will not participate in peer/halo exchanges
        TreeNodeIndex startIndex = findNodeAbove(focusTree.data(), focusTree.size(), peerSfcStart);
        TreeNodeIndex endIndex   = findNodeBelow(focusTree.data(), focusTree.size(), peerSfcEnd);

        if (endIndex < startIndex) { endIndex = startIndex; }
        focusAssignment[peer] = TreeIndexPair(startIndex, endIndex);
    }

    KeyType startKey = domainTree[assignment.firstNodeIdx(myRank)];
    KeyType endKey   = domainTree[assignment.lastNodeIdx(myRank)];

    TreeNodeIndex newStartIndex = findNodeAbove(focusTree.data(), focusTree.size(), startKey);
    TreeNodeIndex newEndIndex   = findNodeBelow(focusTree.data(), focusTree.size(), endKey);
    focusAssignment[myRank]     = TreeIndexPair(newStartIndex, newEndIndex);
}

/*! @brief Based on global assignment, create the list of local particle index ranges to send to each rank
 *
 * @tparam KeyType      32- or 64-bit integer
 * @param assignment    global space curve assignment to ranks
 * @param tree          global cornerstone octree that matches the node counts used to create @p assignment
 * @param particleKeys  sorted list of SFC keys of local particles present on this rank
 * @return              for each rank, a list of index ranges into @p particleKeys to send
 *
 * Converts the global assignment particle keys ranges into particle indices with binary search
 */
template<class KeyType>
SendRanges createSendRanges(const SpaceCurveAssignment& assignment,
                            gsl::span<const KeyType> treeLeaves,
                            gsl::span<const KeyType> particleKeys)
{
    int numRanks = assignment.numRanks();

    SendRanges ret(numRanks + 1);
    for (int rank = 0; rank < numRanks; ++rank)
    {
        KeyType rangeStart = treeLeaves[assignment.firstNodeIdx(rank)];
        ret[rank] = std::lower_bound(particleKeys.begin(), particleKeys.end(), rangeStart) - particleKeys.begin();
    }
    ret.back() = particleKeys.size();

    return ret;
}

/*! @brief return @p numRanks equal length SFC segments for initial domain decomposition
 *
 * @tparam KeyType
 * @param numRanks    number of segments
 * @param level       maximum tree depths or (=number of non-zero leading octal digits)
 * @return            the segments
 *
 * Example: returns [0 2525200000 5252500000 10000000000] for numRanks = 3 and level = 5
 */
template<class KeyType>
std::vector<KeyType> initialDomainSplits(int numRanks, int level)
{
    std::vector<KeyType> ret(numRanks + 1);
    KeyType delta = nodeRange<KeyType>(0) / numRanks;

    ret.front() = 0;
    for (int i = 1; i < numRanks; ++i)
    {
        ret[i] = enclosingBoxCode(KeyType(i) * delta, level);
    }
    ret.back() = nodeRange<KeyType>(0);

    return ret;
}

} // namespace cstone
