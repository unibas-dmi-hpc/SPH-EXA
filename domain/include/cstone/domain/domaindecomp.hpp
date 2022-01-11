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
#include <vector>

#include "cstone/primitives/gather.hpp"
#include "cstone/tree/octree.hpp"
#include "cstone/util/index_ranges.hpp"
#include "cstone/util/gsl-lite.hpp"


namespace cstone
{

/*! @brief a custom type for type safety in function calls
 *
 * The resulting type behaves like an int, except that explicit
 * conversion is required in function calls. Used e.g. in
 * SpaceCurveAssignment::addRange to force the caller to write
 * addRange(Rank(r), a,b,c) instead of addRange(r,a,b,c).
 * This makes it impossible to unintentionally mix up the arguments.
 */
using Rank = StrongType<int, struct RankTag>;

/*! @brief stores which parts of the SFC belong to which rank, on a per-rank basis
 *
 * @tparam I  32- or 64-bit unsigned integer
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
    SpaceCurveAssignment() = default;

    explicit SpaceCurveAssignment(int numRanks) : rankAssignment_(numRanks+1, untouched), counts_(numRanks) {}

    //! @brief add an index/code range to rank @p rank
    void addRange(Rank rank, TreeNodeIndex lower, TreeNodeIndex upper, std::size_t cnt)
    {
        // make sure that there's no holes or overlap between or with the range of the previous rank
        assert(rankAssignment_[rank] == lower || rankAssignment_[rank] == untouched);

        rankAssignment_[rank]   = lower;
        // will be overwritten by @p lower of rank+1, except if rank == numRanks-1
        rankAssignment_[rank+1] = upper;
        counts_[rank]           = cnt;
    }

    [[nodiscard]] int numRanks() const { return int(rankAssignment_.size()) - 1; }

    [[nodiscard]] TreeNodeIndex firstNodeIdx(int rank) const
    {
        return rankAssignment_[rank];
    }

    [[nodiscard]] TreeNodeIndex lastNodeIdx(int rank) const
    {
        return rankAssignment_[rank+1];
    }

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
    std::vector<size_t>        counts_;
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
inline
SpaceCurveAssignment singleRangeSfcSplit(const std::vector<unsigned>& globalCounts, int nSplits)
{
    // one element per rank
    SpaceCurveAssignment ret(nSplits);

    std::size_t globalNParticles = std::accumulate(begin(globalCounts), end(globalCounts), std::size_t(0));

    // distribute work, every rank gets global count / nSplits,
    // the remainder gets distributed one by one
    std::vector<std::size_t> nParticlesPerSplit(nSplits, globalNParticles/nSplits);
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
            if (targetCount < splitCount + globalCounts[j] && // overshoot
                targetCount - splitCount < splitCount + globalCounts[j] - targetCount) // overshoot more than undershoot
            { break; }

            splitCount += globalCounts[j++];
        }

        if (split < nSplits - 1)
        {
            // carry over difference of particles over/under assigned to next split
            // to avoid accumulating round off
            long int delta = (long int)(targetCount) - (long int)(splitCount);
            nParticlesPerSplit[split+1] += delta;
        }
        // afaict, j < nNodes(globalTree) can only happen if there are empty nodes at the end
        else {
            for( ; j < TreeNodeIndex(globalCounts.size()); ++j)
                splitCount += globalCounts[j];
        }

        // other distribution strategies might have more than one range per rank
        ret.addRange(Rank(split), leavesDone, j, splitCount);
        leavesDone = j;
    }

    return ret;
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
    for (int peer : peerRanks)
    {
        KeyType peerSfcStart = domainTree[assignment.firstNodeIdx(peer)];
        KeyType peerSfcEnd   = domainTree[assignment.lastNodeIdx(peer)];

        // Note: start-end range is narrowed down if no exact match is found.
        // the discarded part will not participate in peer/halo exchanges
        TreeNodeIndex startIndex = findNodeAbove(focusTree, peerSfcStart);
        TreeNodeIndex endIndex   = findNodeBelow(focusTree, peerSfcEnd);

        if (endIndex < startIndex) { endIndex = startIndex; }
        focusAssignment[peer] = TreeIndexPair(startIndex, endIndex);
    }

    KeyType startKey = domainTree[assignment.firstNodeIdx(myRank)];
    KeyType endKey   = domainTree[assignment.lastNodeIdx(myRank)];

    TreeNodeIndex newStartIndex = findNodeAbove(focusTree, startKey);
    TreeNodeIndex newEndIndex   = findNodeBelow(focusTree, endKey);
    focusAssignment[myRank] = TreeIndexPair(newStartIndex, newEndIndex);
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
SendList createSendList(const SpaceCurveAssignment& assignment,
                        gsl::span<const KeyType> treeLeaves,
                        gsl::span<const KeyType> particleKeys)
{
    using IndexType = SendManifest::IndexType;
    int nRanks      = assignment.numRanks();

    SendList ret(nRanks);

    for (int rank = 0; rank < nRanks; ++rank)
    {
        SendManifest& manifest = ret[rank];

        KeyType rangeStart = treeLeaves[assignment.firstNodeIdx(rank)];
        KeyType rangeEnd   = treeLeaves[assignment.lastNodeIdx(rank)];

        auto lit = std::lower_bound(particleKeys.begin(), particleKeys.end(), rangeStart);
        IndexType lowerParticleIndex = std::distance(particleKeys.begin(), lit);

        auto uit = std::lower_bound(particleKeys.begin() + lowerParticleIndex, particleKeys.end(), rangeEnd);
        IndexType upperParticleIndex = std::distance(particleKeys.begin(), uit);

        manifest.addRange(lowerParticleIndex, upperParticleIndex);
    }

    return ret;
}

/*! @brief extract elements from the source array through the ordering
 *
 * @tparam T           float or double
 * @param manifest     contains the index ranges of @p source to put into the send buffer
 * @param source       e.g. x,y,z,h arrays
 * @param ordering     the space curve ordering to handle unsorted source arrays
 *                     if source is space-curve-sorted, @p ordering is the trivial 0,1,...,n sequence
 * @param destination  write buffer for extracted elements
 */
template<class T, class IndexType>
void extractRange(const SendManifest& manifest, const T* source, const IndexType* ordering, T* destination)
{
    int idx = 0;
    for (std::size_t rangeIndex = 0; rangeIndex < manifest.nRanges(); ++rangeIndex)
        for (IndexType i = manifest.rangeStart(rangeIndex); i < IndexType(manifest.rangeEnd(rangeIndex)); ++i)
            destination[idx++] = source[ordering[i]];
}

} // namespace cstone
