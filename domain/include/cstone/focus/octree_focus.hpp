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
#include "cstone/tree/octree_internal_td.hpp"
#include "cstone/traversal/traversal.hpp"

namespace cstone
{

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

        TreeNodeIndex firstSibling = leafIdx - siblingIdx;
        TreeNodeIndex lastSibling  = firstSibling + 8;

        // inFringe: leadIdx not in focus, but at least one sibling is in the focus
        // in that case we cannot remove the nodes based on a MAC criterion
        bool inFringe = overlapTwoRanges(firstSibling, lastSibling, firstFocusNode, lastFocusNode);

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
 * @param[in] leafCounts       particle counts per leaf node, length = @p numLeafNodes
 * @param[in] macs             multipole pass or fail per node, length = @p numInternalNodes + numLeafNodes
 * @param[in] firstFocusNode   first focus node in @p cstoneTree, range = [0:numLeafNodes]
 * @param[in] lastFocusNode    last focus node in @p cstoneTree, range = [0:numLeafNodes]
 * @param[in] bucketSize       maximum particle count per (leaf) node and
 *                             minimum particle count (strictly >) for (implicit) internal nodes
 * @param[out] nodeOps         stores rebalance decision result for each node, length = @p numLeafNodes()
 * @return                     true if converged, false otherwise
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

/*! @brief combines the particle count and multipole criteria for rebalancing
 *
 * @return
 *      - 0 if node to be merged
 *      - 1 if node to stay unchanged
 *      - 8 if node to be split
 */
template<class KeyType>
inline HOST_DEVICE_FUN int mergeCountAndMacOpTd(TreeNodeIndex nodeIdx,
                                                const KeyType* nodeKeys,
                                                const TreeNodeIndex* childOffsets,
                                                const TreeNodeIndex* parents,
                                                const unsigned* counts,
                                                const char* macs,
                                                KeyType focusStart,
                                                KeyType focusEnd,
                                                unsigned bucketSize)
{
    TreeNodeIndex siblingGroup = (nodeIdx - 1) / 8;
    TreeNodeIndex parent = nodeIdx ? parents[siblingGroup] : 0;

    TreeNodeIndex firstSibling = childOffsets[parent];
    auto g = childOffsets + firstSibling;
    bool onlyLeafSiblings = (g[0]+g[1]+g[2]+g[3]+g[4]+g[5]+g[6]+g[7]) == 0;

    TreeNodeIndex siblingIdx = nodeIdx - firstSibling;
    KeyType nodeKey          = decodePlaceholderBit(nodeKeys[nodeIdx]);
    unsigned level           = decodePrefixLength(nodeKeys[nodeIdx]) / 3;

    if (onlyLeafSiblings && siblingIdx > 0)
    {
        bool countMerge = counts[parent] <= bucketSize;
        bool macMerge   = macs[parent] == 0;

        KeyType firstGroupKey = decodePlaceholderBit(nodeKeys[firstSibling]);
        KeyType lastGroupKey  = firstGroupKey + 8 * nodeRange<KeyType>(level);
        // inFringe: nodeIdx not in focus, but at least one sibling is in the focus
        // in that case we cannot remove the nodes based on a MAC criterion
        bool inFringe = overlapTwoRanges(firstGroupKey, lastGroupKey, focusStart, focusEnd);

        if (countMerge || (macMerge && !inFringe)) { return 0; } // merge
    }

    bool inFocus = (nodeKey >= focusStart && nodeKey < focusEnd);
    if (level < maxTreeLevel<KeyType>{} && counts[nodeIdx] > bucketSize && (macs[nodeIdx] || inFocus))
    {
        return 8; // split
    }

    return 1; // default: do nothing
}

/*! @brief Compute locally essential split or fuse decision for each octree node in parallel
 *
 * @tparam    KeyType          32- or 64-bit unsigned integer type
 * @param[in] nodeKeys         warren-salmon SFC keys for each tree node, length = @p numNodes
 * @param[in] childOffsets     node index of first child (0 identifies a leaf), length = @p numNodes
 * @param[in] parents          parent node index for each group of 8 siblings, length = (numNodes-1) / 8
 * @param[in] counts           particle count of each tree node, length = @p numNodes
 * @param[in] macs             multipole pass or fail per node, length = @p numNodes
 * @param[in] focusStart       first focus SFC key
 * @param[in] focusEnd         last focus SFC key
 * @param[in] bucketSize       maximum particle count per (leaf) node and
 *                             minimum particle count (strictly >) for (implicit) internal nodes
 * @param[out] nodeOps         stores rebalance decision result for each node, length = @p numNodes
 *                             only leaf nodes will be set, internal nodes are ignored
 * @return                     true if converged, false otherwise
 *
 * For each leaf node i in the tree, in nodeOps[i], stores
 *  - 0 if to be merged
 *  - 1 if unchanged,
 *  - 8 if to be split.
 */
template<class KeyType, class LocalIndex>
bool rebalanceDecisionEssentialTd(const KeyType* nodeKeys,
                                  const TreeNodeIndex* childOffsets,
                                  const TreeNodeIndex* parents,
                                  const unsigned* counts,
                                  const char* macs,
                                  TreeNodeIndex numNodes,
                                  KeyType focusStart,
                                  KeyType focusEnd,
                                  unsigned bucketSize,
                                  LocalIndex* nodeOps)
{
    bool converged = true;
    #pragma omp parallel
    {
        bool convergedThread = true;
        #pragma omp for
        for (TreeNodeIndex i = 0; i < numNodes; ++i)
        {
            // ignore internal nodes
            if (childOffsets[i] != 0) { continue; }
            int opDecision = mergeCountAndMacOpTd(i, nodeKeys, childOffsets, parents, counts, macs, focusStart,
                                                  focusEnd, bucketSize);
            if (opDecision != 1) { convergedThread = false; }

            nodeOps[i] = opDecision;
        }
        if (!convergedThread) { converged = false; }
    }
    return converged;
}

enum class ResolutionStatus : int
{
    //! @brief required SFC keys present in tree, no action needed
    converged,
    //! @brief required SFC keys already present in tree, but had to cancel rebalance-merge operations
    cancelMerge,
    //! @brief subsequent rebalance can resolve the required SFC key by subdividing the closest node
    rebalance,
    //! @brief subsequent rebalance cannot resolve the required SFC key with subdividing the closest node
    failed
};

/*! @brief  modify nodeOps, such that the input tree will contain all mandatory keys after rebalancing
 *
 * @tparam KeyType                32- or 64-bit unsigned integer type
 * @param[in]    treeLeaves       cornerstone octree leaves
 * @param[in]    mandatoryKeys    sequence of keys that @p treeLeaves has to contain when
 *                                rebalancing with @p nodeOps
 * @param[inout] nodeOps          rebalance op-code sequence for @p treeLeaves
 * @return                        resolution status of @p mandatoryKeys
 *
 * After this procedure is called, newTreeLeaves generated by
 *     rebalanceTree(treeLeaves, newTreeLeaves, nodeOps);
 * will contain all the SFC keys listed in mandatoryKeys.
 */
template<class KeyType>
ResolutionStatus enforceKeys(gsl::span<const KeyType> treeLeaves,
                             gsl::span<const KeyType> mandatoryKeys,
                             gsl::span<TreeNodeIndex> nodeOps)
{
    ResolutionStatus status = ResolutionStatus::converged;

    for (KeyType key : mandatoryKeys)
    {
        if (key == 0 || key == nodeRange<KeyType>(0)) { continue; }

        TreeNodeIndex nodeIdx = findNodeBelow(treeLeaves, key);

        auto [siblingIdx, level] = siblingAndLevel(treeLeaves.data(), nodeIdx);

        bool canCancel = siblingIdx > -1;
        // need to cancel if the closest tree node would be merged or the mandatory key is not there
        bool needToCancel = nodeOps[nodeIdx] == 0 || treeLeaves[nodeIdx] != key;
        if (canCancel && needToCancel)
        {
            status = std::max(status, ResolutionStatus::cancelMerge);
            // pointer to sibling-0 nodeOp
            TreeNodeIndex* g = nodeOps.data() + nodeIdx - siblingIdx;
            for (int octant = 0; octant < 8; ++octant)
            {
                if (g[octant] == 0) { g[octant] = 1; } // cancel node merge
            }
        }

        if (treeLeaves[nodeIdx] != key) // mandatory key is not present
        {
            int keyPos = lastNzPlace(key);

            // only add 1 level, otherwise nodes can be added in a non-peer area,
            // exceeding the resolution of the global tree, which will result in a failure to compute
            // exact particle counts for those nodes
            constexpr int maxAddLevels = 1;
            int levelDiff              = keyPos - level;
            if (levelDiff > maxAddLevels) { status = ResolutionStatus::failed; }
            else
            {
                status = std::max(status, ResolutionStatus::rebalance);
            }

            levelDiff        = std::min(levelDiff, maxAddLevels);
            nodeOps[nodeIdx] = std::max(nodeOps[nodeIdx], 1 << (3 * levelDiff));
        }
    }
    return status;
}

/*! @brief inject specified keys into a cornerstone leaf tree
 *
 * @tparam KeyVector    vector of 32- or 64-bit integer
 * @param[inout] tree   cornerstone octree
 * @param[in]    keys   list of SFC keys to insert
 *
 * This function needs to insert more than just @p keys, due the cornerstone
 * invariant of consecutive nodes always having a power-of-8 difference.
 * This means that each subdividing a node, all 8 children always have to be added.
 */
template<class KeyVector>
void injectKeys(KeyVector& tree, gsl::span<const typename KeyVector::value_type> keys)
{
    using KeyType = typename KeyVector::value_type;

    std::vector<KeyType> spanningKeys(keys.begin(), keys.end());
    spanningKeys.push_back(0);
    spanningKeys.push_back(nodeRange<KeyType>(0));
    std::sort(begin(spanningKeys), end(spanningKeys));
    auto uit = std::unique(begin(spanningKeys), end(spanningKeys));
    spanningKeys.erase(uit, end(spanningKeys));

    // spanningTree is a list of all the missing nodes needed to resolve the mandatory keys
    auto spanningTree = computeSpanningTree<KeyType>(spanningKeys);
    tree.reserve(tree.size() + spanningTree.size());

    // spanningTree is now inserted into newLeaves
    std::copy(begin(spanningTree), end(spanningTree), std::back_inserter(tree));

    // cleanup, restore invariants: sorted-ness, no-duplicates
    std::sort(begin(tree), end(tree));
    uit = std::unique(begin(tree), end(tree));
    tree.erase(uit, end(tree));
}


template<class KeyType>
class FocusedOctreeCore
{
public:
    FocusedOctreeCore(unsigned bucketSize)
        : bucketSize_(bucketSize)
    {
        std::vector<KeyType> init{0, nodeRange<KeyType>(0)};
        tree_.update(init.data(), nNodes(init));
    }

    /*! @brief perform a local update step
     *
     * @param[in] focusStart      start of the focus area
     * @param[in] focusEnd        end of the focus area
     * @param[in] mandatoryKeys   List of SFC keys that have to be present in the focus tree after this function
     *                            returns. @p focusStart and @p focusEnd are always mandatory, so they don't need to be
     *                            specified here. @p mandatoryKeys need not be sorted and can tolerate duplicates.
     *                            This is used e.g. to guarantee that the assignment boundaries of peer ranks are
     *                            resolved, even if the update did not converge.
     * @param[in] counts          leaf particle counts, length = tree_.numLeafNodes()
     * @param[in] macs            MAC pass/fail results for each node, length = tree_.numTreeNodes()
     * @return                    true if the tree structure did not change
     */
    bool update(KeyType focusStart,
                KeyType focusEnd,
                gsl::span<const KeyType> mandatoryKeys,
                gsl::span<const unsigned> counts,
                gsl::span<const char> macs)
    {
        [[maybe_unused]] TreeNodeIndex numNodes = tree_.numTreeNodes();
        assert(TreeNodeIndex(counts.size()) == numNodes);
        assert(TreeNodeIndex(macs.size()) == numNodes);

        //TreeNodeIndex firstFocusNode = findNodeBelow(leaves, focusStart);
        //TreeNodeIndex lastFocusNode  = findNodeAbove(leaves, focusEnd);

        gsl::span<TreeNodeIndex> nodeOps(tree_.binaryToOct_);
        gsl::span<TreeNodeIndex> nodeOpsAll(tree_.nodeOrder_);
        bool converged = rebalanceDecisionEssentialTd(tree_.nodeKeys(), tree_.childOffsets(), tree_.parents(),
                                                      counts.data(), macs.data(), tree_.numTreeNodes(), focusStart,
                                                      focusEnd, bucketSize_, nodeOpsAll.data());
        tree_.template extractLeaves<TreeNodeIndex>(nodeOpsAll, nodeOps);

        std::vector<KeyType> allMandatoryKeys{focusStart, focusEnd};
        std::copy(mandatoryKeys.begin(), mandatoryKeys.end(), std::back_inserter(allMandatoryKeys));

        gsl::span<const KeyType> leaves = tree_.treeLeaves();
        auto status = enforceKeys<KeyType>(leaves, allMandatoryKeys, nodeOps);

        if (status == ResolutionStatus::cancelMerge)
        {
            converged = std::all_of(nodeOps.begin(), nodeOps.end() - 1, [](TreeNodeIndex i) { return i == 1; });
        }
        else if (status == ResolutionStatus::rebalance)
        {
            converged = false;
        }

        auto& newLeaves = tree_.prefixes_;
        rebalanceTree(leaves, newLeaves, nodeOps.data());

        // if rebalancing couldn't introduce the mandatory keys, we force-inject them now into the tree
        if (status == ResolutionStatus::failed)
        {
            converged = false;
            injectKeys(newLeaves, allMandatoryKeys);
        }

        swap(newLeaves, tree_.cstoneTree_);
        tree_.updateInternalTree();

        return converged;
    }

    //! @brief provide access to the linked octree
    const TdOctree<KeyType>& octree() const { return tree_; }

    //! @brief returns a view of the tree leaves
    gsl::span<const KeyType> treeLeaves() const { return tree_.treeLeaves(); }

private:
    //! @brief max number of particles per node in focus
    unsigned bucketSize_;

    //! @brief the focused tree
    TdOctree<KeyType> tree_;
};

/*! @brief A fully traversable octree, locally focused w.r.t a MinMac criterion
 *
 * This single rank version is only useful in unit tests.
 */
template<class KeyType>
class FocusedOctreeSingleNode
{
public:
    FocusedOctreeSingleNode(unsigned bucketSize, float theta)
        : theta_(theta)
        , tree_(bucketSize)
        , counts_{bucketSize + 1}
        , macs_{1}
    {
    }

    //! @brief perform a local update step, see FocusedOctreeCore
    template<class T>
    bool update(const Box<T>& box,
                gsl::span<const KeyType> particleKeys,
                KeyType focusStart,
                KeyType focusEnd,
                gsl::span<const KeyType> mandatoryKeys)
    {
        bool converged = tree_.update(focusStart, focusEnd, mandatoryKeys, counts_, macs_);

        macs_.resize(tree_.octree().numTreeNodes());
        markMac(tree_.octree(), box, focusStart, focusEnd, 1.0 / theta_, macs_.data());

        gsl::span<const KeyType> leaves = tree_.treeLeaves();
        leafCounts_.resize(nNodes(leaves));
        computeNodeCounts(leaves.data(), leafCounts_.data(), nNodes(leaves), particleKeys.data(),
                          particleKeys.data() + particleKeys.size(), std::numeric_limits<unsigned>::max(), true);

        counts_.resize(octree().numTreeNodes());
        upsweepSum<unsigned>(octree(), leafCounts_, counts_);

        return converged;
    }

    const TdOctree<KeyType>& octree() const { return tree_.octree(); }

    gsl::span<const KeyType>  treeLeaves() const { return tree_.treeLeaves(); }
    gsl::span<const unsigned> leafCounts() const { return leafCounts_; }

private:
    //! @brief opening angle refinement criterion
    float theta_;

    FocusedOctreeCore<KeyType> tree_;

    //! @brief particle counts of the focused tree leaves
    std::vector<unsigned> leafCounts_;
    std::vector<unsigned> counts_;
    //! @brief mac evaluation result relative to focus area (pass or fail)
    std::vector<char> macs_;
};

} // namespace cstone
