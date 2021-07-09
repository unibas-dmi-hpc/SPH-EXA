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
 * @brief Utility functions for determining the layout of particle buffers on a given rank
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * Each rank will be assigned a part of the SFC, equating to one or multiple ranges of
 * node indices of the global cornerstone octree. In a addition to the assigned nodes,
 * each rank must also store particle data for those nodes in the global octree which are
 * halos of the assigned nodes. Both types of nodes present on the rank are stored in the same
 * particle array (x,y,z,h,...) according to increasing node index, which is the same
 * as increasing Morton code.
 *
 * Given
 *  - the global cornerstone tree
 *  - its assignment to ranks
 *  - lists of in/outgoing halo nodes (global indices) per rank,
 * the utility functions in this file determine the position and size of each node (halo or assigned node)
 * in the particle buffers. The resulting layout is valid for all particle buffers, such as x,y,z,h,d,p,...
 *
 * Note:
 * If a node of the global cornerstone octree has index i, this means its Morton code range is tree[i] - tree[i+1]
 */

#pragma once

#include <vector>

#include "cstone/domain/domaindecomp.hpp"
#include "cstone/halos/discovery.hpp"

namespace cstone
{

/*! @brief Compute send/receive node lists from halo pair node indices
 *
 * @param[in]  assignment       stores which rank owns which part of the SFC
 * @param[in]  haloPairs        list of mutually overlapping pairs of local/remote nodes
 * @param[out] incomingNodes    sorted list of halo nodes to be received,
 *                              grouped by source rank
 * @param[out] outgoingNodes    sorted list of internal nodes to be sent,
 *                              grouped by destination rank
 */
inline
void computeSendRecvNodeList(const SpaceCurveAssignment& assignment,
                             const std::vector<pair<TreeNodeIndex>>& haloPairs,
                             std::vector<std::vector<TreeNodeIndex>>& incomingNodes,
                             std::vector<std::vector<TreeNodeIndex>>& outgoingNodes)
{
    incomingNodes.resize(assignment.numRanks());
    outgoingNodes.resize(assignment.numRanks());

    for (auto& p : haloPairs)
    {
        // as defined in findHalos, the internal node index is stored first
        TreeNodeIndex internalNodeIdx = p[0];
        TreeNodeIndex remoteNodeIdx   = p[1];

        int remoteRank = assignment.findRank(remoteNodeIdx);

        incomingNodes[remoteRank].push_back(remoteNodeIdx);
        outgoingNodes[remoteRank].push_back(internalNodeIdx);
    }

    // remove duplicates in receiver list
    for (auto& v : incomingNodes)
    {
        std::sort(begin(v), end(v));
        auto unique_end = std::unique(begin(v), end(v));
        v.erase(unique_end, end(v));
    }

    // remove duplicates in sender list
    for (auto& v : outgoingNodes)
    {
        std::sort(begin(v), end(v));
        auto unique_end = std::unique(begin(v), end(v));
        v.erase(unique_end, end(v));
    }
}

//! @brief create a sorted list of nodes from the hierarchical per rank node list
inline std::vector<TreeNodeIndex> flattenNodeList(const std::vector<std::vector<TreeNodeIndex>>& groupedNodes)
{
    TreeNodeIndex nNodes = 0;
    for (auto& v : groupedNodes) nNodes += v.size();

    std::vector<TreeNodeIndex> nodeList;
    nodeList.reserve(nNodes);

    // add all halos to nodeList
    for (const auto& group : groupedNodes)
    {
        std::copy(begin(group), end(group), std::back_inserter(nodeList));
    }

    return nodeList;
}

/*! @brief computes the array layout for particle buffers of the executing rank
 *
 * @param[in]  firstLocalNode     First tree node index assigned to executing rank
 * @param[in]  lastLocalNode      Last tree node index assigned to executing rank
 * @param[in]  haloNodes          List of halo node indices without duplicates.
 *                                From the perspective of the
 *                                executing rank, these are incoming halo nodes.
 * @param[in]  globalNodeCounts   Particle count per node in the global octree
 * @param[out] presentNodes       Upon return, will contain a sorted list of global node indices
 *                                present on the executing rank
 * @param[out] offsets            Will contain an offset index for each node in @p presentNodes,
 *                                indicating its position in the particle x,y,z,... buffers,
 *                                length @p presentNodes.size() + 1
 */
template<class IndexType>
static void computeLayoutOffsets(TreeNodeIndex firstLocalNode,
                                 TreeNodeIndex lastLocalNode,
                                 const std::vector<TreeNodeIndex>& haloNodes,
                                 const std::vector<unsigned>& globalNodeCounts,
                                 std::vector<TreeNodeIndex>& presentNodes,
                                 std::vector<IndexType>& offsets)
{
    // add all halo nodes to present
    std::copy(begin(haloNodes), end(haloNodes), std::back_inserter(presentNodes));

    for (TreeNodeIndex i = firstLocalNode; i < lastLocalNode; ++i)
        presentNodes.push_back(i);

    std::sort(begin(presentNodes), end(presentNodes));

    // an extract of globalNodeCounts, containing only nodes listed in localNodes and incomingHalos
    std::vector<unsigned> nodeCounts(presentNodes.size());

    // extract particle count information for all nodes in nodeList
    for (std::size_t i = 0; i < presentNodes.size(); ++i)
    {
        TreeNodeIndex globalNodeIndex = presentNodes[i];
        nodeCounts[i]                 = globalNodeCounts[globalNodeIndex];
    }

    offsets.resize(nodeCounts.size() + 1);
    stl::exclusive_scan(nodeCounts.begin(), nodeCounts.end(), offsets.begin(), 0);
    offsets.back() = offsets[nodeCounts.size()-1] + nodeCounts.back();
}


/*! @brief translate global node indices involved in halo exchange to array ranges
 *
 * @param haloNodeIndices   For each rank, a list of global node indices to send or receive.
 *                          Each node referenced in these lists must be contained in
 *                          @p presentNodes.
 * @param presentNodes      Sorted unique list of global node indices present on the
 *                          executing rank
 * @param nodeOffsets       nodeOffset[i] stores the location of the node presentNodes[i]
 *                          in the particle arrays. nodeOffset[presentNodes.size()] stores
 *                          the total size of the particle arrays on the executing rank.
 *                          Size is presentNodes.size() + 1
 * @return                  For each rank, the returned sendList has one or multiple ranges of indices
 *                          of local particle arrays to send or receive.
 */
template<class IndexType>
static SendList createHaloExchangeList(const std::vector<std::vector<TreeNodeIndex>>& haloNodeIndices,
                                       const std::vector<TreeNodeIndex>& presentNodes,
                                       const std::vector<IndexType>& nodeOffsets)
{
    SendList sendList(haloNodeIndices.size());

    for (std::size_t rank = 0; rank < sendList.size(); ++rank)
    {
        for (TreeNodeIndex globalNodeIndex : haloNodeIndices[rank])
        {
            TreeNodeIndex localNodeIndex = std::lower_bound(begin(presentNodes), end(presentNodes), globalNodeIndex)
                                            - begin(presentNodes);

            sendList[rank].addRange(nodeOffsets[localNodeIndex], nodeOffsets[localNodeIndex+1]);
        }
    }

    return sendList;
}

/*! @brief calculate the location (offset) of each focus tree leaf node in the particle arrays
 *
 * @param focusLeafCounts   node counts of the focus leaves, size N
 * @param haloFlags         flag for each node, with a non-zero value if present as halo node, size N
 * @param firstAssignedIdx  first focus leaf idx to treat as part of the assigned nodes on the executing rank
 * @param lastAssignedIdx   last focus leaf idx to treat as part of the assigned nodes on the executing rank
 * @return                  array with offsets, size N+1. The first element is zero, the last element is
 *                          equal to the sum of all all present (assigned+halo) node counts.
 */
inline
std::vector<LocalParticleIndex> computeNodeLayout(gsl::span<const unsigned> focusLeafCounts,
                                                  gsl::span<const int> haloFlags,
                                                  TreeNodeIndex firstAssignedIdx,
                                                  TreeNodeIndex lastAssignedIdx)
{
    std::vector<LocalParticleIndex> layout(focusLeafCounts.size() + 1, 0);

    #pragma omp parallel for
    for (TreeNodeIndex i = 0; i < focusLeafCounts.size(); ++i)
    {
        bool onRank = firstAssignedIdx <= i && i < lastAssignedIdx;
        if (onRank || haloFlags[i]) { layout[i] = focusLeafCounts[i]; }
    }

    exclusiveScan(layout.data(), layout.size());

    return layout;
}

/*! @brief computes a list which local array ranges are going to be filled with halo particles
 *
 * @param layout       prefix sum of leaf counts of locally present nodes (see computeNodeLayout)
 *                     length N+1
 * @param haloFlags    0 or 1 for each leaf, length N
 * @param assignment   assignment of leaf nodes to peer ranks
 * @param peerRanks    list of peer ranks
 * @return             list of array index ranges for the receiving part in exchangeHalos
 */
inline
SendList computeHaloReceiveList(gsl::span<const LocalParticleIndex> layout,
                                gsl::span<const int> haloFlags,
                                gsl::span<const TreeIndexPair> assignment,
                                gsl::span<const int> peerRanks)
{
    SendList ret(assignment.size());

    for (int peer : peerRanks)
    {
        TreeNodeIndex peerStartIdx = assignment[peer].start();
        TreeNodeIndex peerEndIdx   = assignment[peer].end();

        std::vector<LocalParticleIndex> receiveRanges =
            extractMarkedElements<LocalParticleIndex>(layout, haloFlags, peerStartIdx, peerEndIdx);

        for (int i = 0; i < receiveRanges.size(); i +=2 )
        {
            ret[peer].addRange(receiveRanges[i], receiveRanges[i+1]);
        }
    }

    return ret;
}

template<class... Arrays>
void relocate(LocalParticleIndex newSize, LocalParticleIndex offset, Arrays&... arrays)
{
    std::array data{(&arrays)...};

    using ArrayType = std::decay_t<decltype(*data[0])>;

    for (auto arrayPtr : data)
    {
        ArrayType newArray(newSize);
        std::copy(arrayPtr->begin(), arrayPtr->end(), newArray.begin() + offset);
        swap(newArray, *arrayPtr);
    }
}

} // namespace cstone
