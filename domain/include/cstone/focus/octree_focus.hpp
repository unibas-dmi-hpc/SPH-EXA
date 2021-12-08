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
#include "cstone/traversal/boxoverlap.hpp"
#include "cstone/util/gsl-lite.hpp"
#include "cstone/util/index_ranges.hpp"

#include "cstone/traversal/macs.hpp"
#include "cstone/tree/octree_internal.hpp"
#include "cstone/traversal/traversal.hpp"

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

/*! @brief combines the particle count and multipole criteria for rebalancing
 *
 * @return
 *      - 0 if node to be merged
 *      - 1 if node to stay unchanged
 *      - 8 if node to be split
 */
template<class KeyType>
inline HOST_DEVICE_FUN
int mergeCountAndMacOp(TreeNodeIndex leafIdx, const KeyType* cstoneTree,
                       TreeNodeIndex numInternalNodes,
                       const TreeNodeIndex* leafParents,
                       const unsigned* leafCounts, const char* macs,
                       TreeNodeIndex firstFocusNode, TreeNodeIndex lastFocusNode,
                       unsigned bucketSize)
{
    auto [siblingIdx, level] = siblingAndLevel(cstoneTree, leafIdx);

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
    if (level < maxTreeLevel<KeyType>{} && leafCounts[leafIdx] > bucketSize
        && (macs[numInternalNodes + leafIdx] || inFocus))
    { return 8; } // split

    return 1; // default: do nothing
}

/*! @brief Compute locally essential split or fuse decision for each octree node in parallel
 *
 * @tparam    KeyType          32- or 64-bit unsigned integer type
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
 * @param[out] nodeOps         stores rebalance decision result for each node, length = @p numLeafNodes()
 * @return                     true if converged, false
 *
 * For each node i in the tree, in nodeOps[i], stores
 *  - 0 if to be merged
 *  - 1 if unchanged,
 *  - 8 if to be split.
 */
template<class KeyType, class LocalIndex>
bool rebalanceDecisionEssential(const KeyType* cstoneTree, TreeNodeIndex numInternalNodes, TreeNodeIndex numLeafNodes,
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

struct NoPeerExchange
{
    template<class KeyType>
    void operator()(gsl::span<const int>, gsl::span<const IndexPair<TreeNodeIndex>>,
                    gsl::span<const KeyType>, gsl::span<const KeyType>, gsl::span<unsigned>)
    {
        throw std::runtime_error("FocusTree without MPI communication cannot perform a global update\n");
    }
};

namespace focused_octree_detail
{
    struct NoCommTag {};
}

template<class, class = void>
struct ExchangePeerCounts
{};

template<class CommunicationType>
struct ExchangePeerCounts<CommunicationType, std::enable_if_t<std::is_same_v<focused_octree_detail::NoCommTag, CommunicationType>>>
{
    using type = NoPeerExchange;
};

template<class CommunicationType>
using ExchangePeerCounts_t = typename ExchangePeerCounts<CommunicationType>::type;


/*! @brief a fully traversable octree with a local focus
 *
 * @tparam KeyType             32- or 64-bit unsigned integer
 * @tparam CommunicationType   NoCommTag or MpiCommTag to enable updateGlobal
 *
 * This class is not intended for direct use. Instead use the type aliases
 * FocusedOctreeSingleNode or FocusedOctree.
 *
 * The focus area can dynamically change.
 */
template<class KeyType, class CommunicationType>
class FocusedOctreeImpl
{
public:

    /*! @brief constructor
     *
     * @param bucketSize    maximum number of particles per leaf inside the focus area
     * @param theta         opening angle parameter for a min-distance MAC criterion
     *                      to determine the adaptive resolution from the focus area.
     *                      In a converged FocusedOctree, each node outside the focus area
     *                      passes the min-distance MAC with theta as the parameter w.r.t
     *                      to any point inside the focus area.
     */
    FocusedOctreeImpl(unsigned bucketSize, float theta)
        : bucketSize_(bucketSize), theta_(theta), counts_{bucketSize+1}, macs_{1}
    {
        tree_.update(std::vector<KeyType>{0, nodeRange<KeyType>(0)});
    }

    /*! @brief perform a local update step
     *
     * @tparam T              float or double
     * @param box             coordinate bounding box
     * @param particleKeys    locally present particle SFC keys
     * @param focusStart      start of the focus area
     * @param focusEnd        end of the focus area
     * @param mandatoryKeys   List of SFC keys that have to be present in the focus tree after this function returns.
     *                        @p focusStart and @p focusEnd are always mandatory, so they don't need to be
     *                        specified here. @p mandatoryKeys need not be sorted and can tolerate duplicates.
     *                        This is used e.g. to guarantee that the assignment boundaries of peer ranks are resolved,
     *                        even if the update did not converge.
     * @return                true if the tree structure did not change
     *
     * First rebalances the tree based on previous node counts and MAC evaluations,
     * then updates the node counts and MACs.
     */
    template<class T>
    bool update(const Box<T>& box, gsl::span<const KeyType> particleKeys, KeyType focusStart, KeyType focusEnd,
                gsl::span<const KeyType> mandatoryKeys)
    {
        assert(std::is_sorted(particleKeys.begin(), particleKeys.end()));

        gsl::span<const KeyType> leaves = tree_.treeLeaves();

        TreeNodeIndex firstFocusNode = findNodeBelow(leaves, focusStart);
        TreeNodeIndex lastFocusNode  = findNodeAbove(leaves, focusEnd);

        std::vector<TreeNodeIndex> nodeOps(tree_.numLeafNodes() + 1);
        bool converged = rebalanceDecisionEssential(leaves.data(), tree_.numInternalNodes(), tree_.numLeafNodes(), tree_.leafParents(),
                                                    counts_.data(), macs_.data(), firstFocusNode, lastFocusNode,
                                                    bucketSize_, nodeOps.data());

        std::vector<KeyType> allMandatoryKeys{focusStart, focusEnd};
        std::copy(mandatoryKeys.begin(), mandatoryKeys.end(), std::back_inserter(allMandatoryKeys));

        auto status = enforceKeys<KeyType>(leaves, allMandatoryKeys, nodeOps);

        if (status == ResolutionStatus::cancelMerge)
        {
            converged = std::all_of(begin(nodeOps), end(nodeOps) -1, [](TreeNodeIndex i) { return i == 1; });
        }
        else if (status == ResolutionStatus::rebalance)
        {
            converged = false;
        }

        std::vector<KeyType> newLeaves;
        rebalanceTree(leaves, newLeaves, nodeOps.data());

        // if rebalancing couldn't introduce the mandatory keys, we force-inject them now into the tree
        if (status == ResolutionStatus::failed)
        {
            converged = false;
            injectKeys(newLeaves, allMandatoryKeys);
        }

        tree_.update(std::move(newLeaves));

        // tree update invalidates the view, need to update it
        leaves = tree_.treeLeaves();

        macs_.resize(tree_.numTreeNodes());
        markMac(tree_, box, focusStart, focusEnd, 1.0 / theta_, macs_.data());

        counts_.resize(tree_.numLeafNodes());
        // local node counts
        computeNodeCounts(leaves.data(), counts_.data(), nNodes(leaves), particleKeys.data(),
                          particleKeys.data() + particleKeys.size(), std::numeric_limits<unsigned>::max(), true);

        return converged;
    }

    /*! @brief perform a global update of the tree structure
     *
     * @tparam T                  float or double
     * @param box                 global coordinate bounding box
     * @param particleKeys        SFC keys of local particles
     * @param myRank              ID of the executing rank
     * @param peerRanks           list of ranks that have nodes that fail the MAC criterion
     *                            w.r.t to the assigned SFC part of @p myRank
     *                            use e.g. findPeersMac to calculate this list
     * @param assignment          assignment of the global leaf tree to ranks
     * @param globalTreeLeaves    global cornerstone leaf tree
     * @param globalCounts        global cornerstone leaf tree counts
     * @return                    true if the tree structure did not change
     *
     * The part of the SFC that is assigned to @p myRank is considered as the focus area.
     *
     * Preconditions:
     *  - The provided assignment and globalTreeLeaves are the same as what was used for
     *    calculating the list of peer ranks with findPeersMac. (not checked)
     *  - All local particle keys must lie within the assignment of @p myRank (checked)
     *    and must be sorted in ascending order (checked)
     *
     * The global update first performs a local update, then adds the node counts from peer ranks and the
     * global tree on top.
     */
    template<class T>
    bool updateGlobal(const Box<T>& box, gsl::span<const KeyType> particleKeys, int myRank, gsl::span<const int> peerRanks,
                      const SpaceCurveAssignment& assignment, gsl::span<const KeyType> globalTreeLeaves,
                      gsl::span<const unsigned> globalCounts)
    {
        KeyType focusStart = globalTreeLeaves[assignment.firstNodeIdx(myRank)];
        KeyType focusEnd   = globalTreeLeaves[assignment.lastNodeIdx(myRank)];

        assert(particleKeys.front() >= focusStart && particleKeys.back() < focusEnd);

        std::vector<KeyType> peerBoundaries;
        for (int peer : peerRanks)
        {
            peerBoundaries.push_back(globalTreeLeaves[assignment.firstNodeIdx(peer)]);
            peerBoundaries.push_back(globalTreeLeaves[assignment.lastNodeIdx(peer)]);
        }

        bool converged = update(box, particleKeys, focusStart, focusEnd, peerBoundaries);

        auto requestIndices = findRequestIndices(peerRanks, assignment, globalTreeLeaves, tree_.treeLeaves());

        ExchangePeerCounts_t<CommunicationType>{}.template operator()<KeyType>
            (peerRanks, requestIndices, particleKeys, tree_.treeLeaves(), counts_);

        TreeNodeIndex firstFocusNode = findNodeAbove(treeLeaves(), focusStart);
        TreeNodeIndex lastFocusNode  = findNodeBelow(treeLeaves(), focusEnd);

        // particle counts for leaf nodes in treeLeaves() / leafCounts():
        //   Node indices [firstFocusNode:lastFocusNode] got assigned counts from local particles.
        //   Node index ranges listed in requestIndices got assigned counts from peer ranks.
        //   All remaining indices need to get their counts from the global tree.
        //   They are stored in globalCountIndices.

        requestIndices.emplace_back(firstFocusNode, lastFocusNode);
        std::sort(requestIndices.begin(), requestIndices.end());
        auto globalCountIndices = invertRanges(0, requestIndices, tree_.numLeafNodes());

        for (auto ip : globalCountIndices)
        {
            countRequestParticles(globalTreeLeaves, globalCounts, treeLeaves().subspan(ip.start(), ip.count() + 1),
                                  gsl::span<unsigned>(counts_.data() + ip.start(), ip.count()));
        }

        return converged;
    }

    //! @brief provide access to the linked octree
    const Octree<KeyType>& octree() const { return tree_; }

    //! @brief returns a view of the tree leaves
    gsl::span<const KeyType> treeLeaves() const { return tree_.treeLeaves(); }

    //! @brief returns a view of the leaf particle counts
    [[nodiscard]] gsl::span<const unsigned> leafCounts() const { return counts_; }

private:

    //! @brief max number of particles per node in focus
    unsigned bucketSize_;
    //! @brief opening angle refinement criterion
    float theta_;

    //! @brief the focused tree
    Octree<KeyType> tree_;
    //! @brief particle counts of the focused tree leaves
    std::vector<unsigned> counts_;
    //! @brief mac evaluation result relative to focus area (pass or fail)
    std::vector<char> macs_;
};

//! @brief Focused octree type for use without MPI (e.g. in unit tests)
template<class KeyType>
using FocusedOctreeSingleNode = FocusedOctreeImpl<KeyType, focused_octree_detail::NoCommTag>;

} // namespace cstone