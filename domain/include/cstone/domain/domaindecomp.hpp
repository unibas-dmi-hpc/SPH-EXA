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

#include "cstone/tree/octree.hpp"
#include "cstone/util.hpp"
#include "cstone/primitives/gather.hpp"


namespace cstone
{

/*! @brief Stores ranges of local particles to be sent to another rank
 *
 * @tparam I  32- or 64-bit signed or unsigned integer to store the indices
 *
 *  Used for SendRanges with index ranges referencing elements in e.g. x,y,z,h arrays.
 */
template<class Index>
class IndexRanges
{
public:
    using IndexType = Index;

    IndexRanges() : totalCount_(0), ranges_{} {}

    //! @brief add a local index range, infer count from difference
    void addRange(IndexType lower, IndexType upper)
    {
        assert(lower <= upper);
        ranges_.emplace_back(lower, upper);
        totalCount_ += upper - lower;
    }

    [[nodiscard]] IndexType rangeStart(size_t i) const
    {
        return ranges_[i][0];
    }

    [[nodiscard]] IndexType rangeEnd(size_t i) const
    {
        return ranges_[i][1];
    }

    //! @brief the number of particles in range i
    [[nodiscard]] std::size_t count(size_t i) const { return ranges_[i][1] - ranges_[i][0]; }

    //! @brief the sum of number of particles in all ranges or total send count
    [[nodiscard]] std::size_t totalCount() const { return totalCount_; }

    [[nodiscard]] std::size_t nRanges() const { return ranges_.size(); }

private:
    friend bool operator==(const IndexRanges& lhs, const IndexRanges& rhs)
    {
        return lhs.totalCount_ == rhs.totalCount_ && lhs.ranges_ == rhs.ranges_;
    }

    std::size_t totalCount_;
    std::vector<pair<IndexType>> ranges_;
};


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
 * The storage layout allows fast look-up of the Morton code ranges that a given rank
 * was assigned.
 *
 * Note: Assignment of SFC ranges to ranks should be unique, each SFC range should only
 * be assigned to one rank. This is NOT checked.
 */
class SpaceCurveAssignment
{
public:
    SpaceCurveAssignment() = default;

    explicit SpaceCurveAssignment(int nRanks) : rankAssignment_(nRanks+1), counts_(nRanks+1) {}

    //! @brief add an index/code range to rank @p rank
    void addRange(Rank rank, TreeNodeIndex lower, TreeNodeIndex upper, std::size_t cnt)
    {
        rankAssignment_[rank]   = lower;
        // will be overwritten by @p lower of rank+1, except if rank == nRanks-1
        rankAssignment_[rank+1] = upper;
        counts_[rank]           = cnt;
    }

    [[nodiscard]] int nRanks() const { return int(rankAssignment_.size()); }

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
 * @param nSplits            divide the global tree into nSplits pieces, sensible choice e.g.: nSplits == nRanks
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
        while (splitCount < targetCount && j < globalCounts.size())
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
            for( ; j < globalCounts.size(); ++j)
                splitCount += globalCounts[j];
        }

        // other distribution strategies might have more than one range per rank
        ret.addRange(Rank(split), leavesDone, j, splitCount);
        leavesDone = j;
    }

    return ret;
}

//! @brief stores one or multiple index ranges of local particles to send out to another rank
using SendManifest = IndexRanges<unsigned>; // works if there are < 2^32 local particles
//! @brief SendList will contain one manifest per rank
using SendList     = std::vector<SendManifest>;

/*! @brief Based on global assignment, create the list of local particle index ranges to send to each rank
 *
 * @tparam I            32- or 64-bit integer
 * @param assignment    global space curve assignment to ranks
 * @param tree          global cornerstone octree that matches the node counts used to create @p assignment
 * @param codesStart    sorted list of morton codes of local particles present on this rank
 * @param codesEnd
 * @return              for each rank, a list of index ranges into @p mortonCodes to send
 *
 * Converts the global assignment Morton code ranges into particle indices with binary search
 */
template<class I>
SendList createSendList(const SpaceCurveAssignment& assignment, const std::vector<I>& tree, const I* codesStart, const I* codesEnd)
{
    using IndexType = SendManifest::IndexType;
    int nRanks      = assignment.nRanks();

    SendList ret(nRanks);

    for (int rank = 0; rank < nRanks; ++rank)
    {
        SendManifest& manifest = ret[rank];

        I rangeStart = tree[assignment.firstNodeIdx(rank)];
        I rangeEnd   = tree[assignment.lastNodeIdx(rank)];

        auto lit = std::lower_bound(codesStart, codesEnd, rangeStart);
        IndexType lowerParticleIndex = std::distance(codesStart, lit);

        auto uit = std::lower_bound(codesStart + lowerParticleIndex, codesEnd, rangeEnd);
        IndexType upperParticleIndex = std::distance(codesStart, uit);

        manifest.addRange(lowerParticleIndex, upperParticleIndex);
    }

    return ret;
}

template<class T, class IndexType>
void extractRange(const SendManifest& manifest, const T* source, const IndexType* ordering, T* destination)
{
    int idx = 0;
    for (std::size_t rangeIndex = 0; rangeIndex < manifest.nRanges(); ++rangeIndex)
        for (IndexType i = manifest.rangeStart(rangeIndex); i < manifest.rangeEnd(rangeIndex); ++i)
            destination[idx++] = source[ordering[i]];
}

/*! @brief create a buffer of elements to send by extracting elements from the source array
 *
 * @tparam T         float or double
 * @param manifest   contains the index ranges of @p source to put into the send buffer
 * @param source     e.g. x,y,z,h arrays
 * @param ordering   the space curve ordering to handle unsorted source arrays
 *                   if source is space-curve-sorted, @p ordering is the trivial 0,1,...,n sequence
 * @return           the send buffer
 */
template<class T, class IndexType>
std::vector<T> createSendBuffer(const SendManifest& manifest, const T* source,
                                const IndexType* ordering)
{
    IndexType sendSize = manifest.totalCount();

    std::vector<T> sendBuffer(sendSize);
    extractRange(manifest, source, ordering, sendBuffer.data());

    return sendBuffer;
}


} // namespace cstone
