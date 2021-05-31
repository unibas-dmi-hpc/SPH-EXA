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
 * @brief Generation of locally essential global octrees in cornerstone format
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * A locally essential octree has a certain global resolution specified by a maximum
 * particle count per leaf node. In addition, it features a focus area defined as a
 * sub-range of the global space filling curve. In this focus sub-range, the resolution
 * can be higher, expressed through a smaller maximum particle count per leaf node.
 * Crucially, the resolution is also higher in the halo-areas of the focus sub-range.
 * These halo-areas can be defined as the overlap with the smoothing-length spheres around
 * the contained particles in the focus sub-range (SPH) or as the nodes whose opening angle
 * is too big to satisfy a multipole acceptance criterion from any perspective within the
 * focus sub-range (N-body).
 */

#pragma once

#include <vector>

#include "cstone/domain/domaindecomp.hpp"
#include "cstone/halos/boxoverlap.hpp"
#include "cstone/util/gsl-lite.hpp"

#include "macs.hpp"
#include "octree_internal.hpp"
#include "traversal.hpp"

namespace cstone
{

/*! @brief determine indices in local focus tree to send out to peer ranks
 *
 * @tparam I                32- or  64-bit unsigned integer
 * @param peers             list of peer ranks (for point-to-point communication)
 * @param assignment        assignment of the global SFC to ranks
 * @param domainTreeLeaves  tree leaves of the global tree in cornerstone format
 * @param focusTreeLeaves   tree leaves of the locally focused tree in cornerstone format
 * @return                  a list of pairs (startIdx, endIdx) referring to elements of @p focusTreeLeaves
 *                          to send out to each peer rank. One of the rare cases where the range has to include
 *                          the last element. To describe n tree nodes, we need n+1 SFC keys.
 */
template<class I>
std::vector<pair<TreeNodeIndex>> findPeerFocusLeaves(gsl::span<const int> peers, const SpaceCurveAssignment& assignment,
                                                     gsl::span<const I> domainTreeLeaves, gsl::span<const I> focusTreeLeaves)
{
    std::vector<pair<TreeNodeIndex>> ret;
    ret.reserve(peers.size());
    for (int peer : peers)
    {
        I peerSfcStart = domainTreeLeaves[assignment.firstNodeIdx(peer)];
        I peerSfcEnd   = domainTreeLeaves[assignment.lastNodeIdx(peer)];

        TreeNodeIndex firstFocusIdx = std::lower_bound(begin(focusTreeLeaves), end(focusTreeLeaves), peerSfcStart)
                                     - begin(focusTreeLeaves);
        TreeNodeIndex lastFocusIdx  = std::lower_bound(begin(focusTreeLeaves), end(focusTreeLeaves), peerSfcEnd)
                                     - begin(focusTreeLeaves);
        ret.emplace_back(firstFocusIdx, lastFocusIdx);
    }

    return ret;
}

template<class I>
inline CUDA_HOST_DEVICE_FUN
int mergeCountAndMacOp(TreeNodeIndex leafIdx, const I* cstoneTree,
                       TreeNodeIndex numInternalNodes,
                       const TreeNodeIndex* leafParents,
                       const unsigned* leafCounts, const char* macs,
                       TreeNodeIndex firstFocusNode, TreeNodeIndex lastFocusNode,
                       unsigned bucketSize)
{
    auto p = siblingAndLevel(cstoneTree, leafIdx);
    unsigned siblingIdx = p[0];
    unsigned level      = p[1];

    if (siblingIdx > 0) // 8 siblings next to each other, node can potentially be merged
    {
        // pointer to first node in sibling group
        auto g = leafCounts + leafIdx - siblingIdx;

        bool countMerge = (g[0]+g[1]+g[2]+g[3]+g[4]+g[5]+g[6]+g[7]) <= bucketSize;
        bool macMerge   = macs[leafParents[leafIdx]] == 0;
        bool inFringe   = leafIdx - siblingIdx + 8 >= firstFocusNode && leafIdx - siblingIdx < lastFocusNode;

        if (countMerge || (macMerge && !inFringe)) { return 0; } // merge
    }

    bool inFocus  = (leafIdx >= firstFocusNode && leafIdx < lastFocusNode);
    if (level < maxTreeLevel<I>{} && leafCounts[leafIdx] > bucketSize
        && (macs[numInternalNodes + leafIdx] || inFocus))
    { return 8; } // split

    return 1; // default: do nothing
}

/*! @brief Compute locally essential split or fuse decision for each octree node in parallel
 *
 * @tparam I                   32- or 64-bit unsigned integer type
 * @param[in] cstoneTree       cornerstone octree leaves, length = @p numLeafNodes
 * @param[in] numInternalNodes number of internal octree nodes
 * @param[in] numLeafNodes     number of leaf octree nodes
 * @param[in] leafParents      stores the parent node index of each leaf, length = @p numLeafNodes
 * @param[in] leafCounts       output particle counts per leaf node, length = @p numLeafNodes
 * @param[in] macs             multipole pass or fail per node, length = @p numInternalNodes + numLeafNodes
 * @param[in] firstFocusNode   first focus node in @p cstoneTree, range = [0:numLeafNodes]
 * @param[in] lastFocusNode    last focus node in @p cstoneTree, range = [0:numLeafNodes]
 * @param[in] bucketSize       maximum particle count per (leaf) node and
 *                             minimum particle count (strictly >) for (implicit) internal nodes
 * @param[out] nodeOps         stores rebalance decision result for each node, length = @p nLeafNodes()
 * @return                     true if converged, false
 *
 * For each node i in the tree, in nodeOps[i], stores
 *  - 0 if to be merged
 *  - 1 if unchanged,
 *  - 8 if to be split.
 */
template<class I, class LocalIndex>
bool rebalanceDecisionEssential(const I* cstoneTree, TreeNodeIndex numInternalNodes, TreeNodeIndex numLeafNodes,
                                const TreeNodeIndex* leafParents,
                                const unsigned* leafCounts, const char* macs,
                                TreeNodeIndex firstFocusNode, TreeNodeIndex lastFocusNode,
                                unsigned bucketSize, LocalIndex* nodeOps)
{
    bool converged = true;
    #pragma omp parallel
    {
        bool convergedThread = true;
        #pragma omp for
        for (TreeNodeIndex leafIdx = 0; leafIdx < numLeafNodes; ++leafIdx)
        {
            int opDecision = mergeCountAndMacOp(leafIdx, cstoneTree, numInternalNodes, leafParents, leafCounts,
                                                macs, firstFocusNode, lastFocusNode, bucketSize);
            if (opDecision != 1) { convergedThread = false; }

            nodeOps[leafIdx] = opDecision;
        }
        if (!convergedThread) { converged = false; }
    }
    return converged;
}

template<class I>
class FocusedOctree
{
public:
    FocusedOctree(unsigned bucketSize, float theta)
        : bucketSize_(bucketSize), theta_(theta), counts_{bucketSize+1}, macs_{1}
    {
        tree_.update(std::vector<I>{0, nodeRange<I>(0)});
    }

    template<class T>
    bool update(const Box<T>& box, const I* codesStart, const I* codesEnd, I focusStart, I focusEnd)
    {
        auto leaves = tree_.treeLeaves();

        TreeNodeIndex firstFocusNode = std::upper_bound(leaves.begin(), leaves.end(), focusStart) - leaves.begin() - 1;
        TreeNodeIndex lastFocusNode  = std::lower_bound(leaves.begin(), leaves.end(), focusEnd) - leaves.begin();

        macs_.resize(tree_.nTreeNodes());
        markMac(tree_, box, focusStart, focusEnd, 1.0/(theta_*theta_), macs_.data());

        std::vector<TreeNodeIndex> nodeOps(tree_.nLeafNodes() + 1);
        bool converged = rebalanceDecisionEssential(leaves.data(), tree_.nInternalNodes(), tree_.nLeafNodes(), tree_.leafParents(),
                                                    counts_.data(), macs_.data(), firstFocusNode, lastFocusNode,
                                                    bucketSize_, nodeOps.data());
        std::vector<I> newLeaves;
        rebalanceTree(leaves, newLeaves, nodeOps.data());
        tree_.update(std::move(newLeaves));
        // update view, because the old one is invalidated by tree_.update()
        leaves = tree_.treeLeaves();

        counts_.resize(tree_.nLeafNodes());
        // local node counts
        computeNodeCounts(leaves.data(), counts_.data(), nNodes(leaves), codesStart, codesEnd,
                          std::numeric_limits<unsigned>::max(), true);

        return converged;
    }

    template<class T>
    bool updateGlobal(const Box<T>& box, const I* codesStart, const I* codesEnd, int myRank, gsl::span<const int> peerRanks,
                      const SpaceCurveAssignment& assignment, gsl::span<const I> globalTreeLeaves)
    {
        bool converged =
        update(box, codesStart, codesEnd, globalTreeLeaves[assignment.firstNodeIdx(myRank)],
               globalTreeLeaves[assignment.lastNodeIdx(myRank)]);

        auto peerFocusLeafIndices = findPeerFocusLeaves(peerRanks, assignment, globalTreeLeaves, tree_.treeLeaves());

        std::vector<I>        tmpLeaves(tree_.nLeafNodes() + 1);
        std::vector<unsigned> tmpCounts(tree_.nLeafNodes());
        exchangeFocus(peerRanks, peerFocusLeafIndices, tree_.treeLeaves(), counts_, tmpLeaves, tmpCounts);

        return converged;
    }

    gsl::span<const I> treeLeaves() const { return tree_.treeLeaves(); }

private:
    //! @brief max number of particles per node in focus
    unsigned bucketSize_;
    //! @brief opening angle refinement criterion
    float theta_;

    //! @brief the focused tree
    Octree<I> tree_;
    //! @brief particle counts of the focused tree leaves
    std::vector<unsigned> counts_;
    //! @brief mac evaluation result relative to focus area (pass or fail)
    std::vector<char> macs_;
};

} // namespace cstone