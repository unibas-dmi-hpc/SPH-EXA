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
 * @brief Focused octree rebalance
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <vector>

#include "cstone/traversal/boxoverlap.hpp"
#include "cstone/tree/csarray.hpp"
#include "cstone/util/gsl-lite.hpp"

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
inline HOST_DEVICE_FUN int mergeCountAndMacOp(TreeNodeIndex nodeIdx,
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
    TreeNodeIndex parent       = nodeIdx ? parents[siblingGroup] : 0;

    TreeNodeIndex firstSibling = childOffsets[parent];
    auto g                     = childOffsets + firstSibling;
    bool onlyLeafSiblings      = (g[0] + g[1] + g[2] + g[3] + g[4] + g[5] + g[6] + g[7]) == 0;

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
template<class KeyType>
bool rebalanceDecisionEssential(gsl::span<const KeyType> nodeKeys,
                                const TreeNodeIndex* childOffsets,
                                const TreeNodeIndex* parents,
                                const unsigned* counts,
                                const char* macs,
                                KeyType focusStart,
                                KeyType focusEnd,
                                unsigned bucketSize,
                                TreeNodeIndex* nodeOps)
{
    bool converged = true;
#pragma omp parallel
    {
        bool convergedThread = true;
#pragma omp for
        for (TreeNodeIndex i = 0; i < nodeKeys.ssize(); ++i)
        {
            // ignore internal nodes
            if (childOffsets[i] != 0) { continue; }
            int opDecision = mergeCountAndMacOp(i, nodeKeys.data(), childOffsets, parents, counts, macs, focusStart,
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

template<class KeyType>
HOST_DEVICE_FUN ResolutionStatus
enforceKeySingle(const KeyType* leaves, TreeNodeIndex* nodeOps, TreeNodeIndex numLeaves, KeyType key)
{
    if (key == 0 || key == nodeRange<KeyType>(0)) { return ResolutionStatus::converged; }

    ResolutionStatus status = ResolutionStatus::converged;

    TreeNodeIndex nodeIdx    = findNodeBelow(leaves, numLeaves + 1, key);
    auto [siblingIdx, level] = siblingAndLevel(leaves, nodeIdx);

    bool canCancel = siblingIdx > -1;
    // need to cancel if the closest tree node would be merged or the mandatory key is not there
    bool needToCancel = nodeOps[nodeIdx] == 0 || leaves[nodeIdx] != key;
    if (canCancel && needToCancel)
    {
        status = ResolutionStatus::cancelMerge;
        // pointer to sibling-0 nodeOp
        TreeNodeIndex* g = nodeOps + nodeIdx - siblingIdx;
        for (int octant = 0; octant < 8; ++octant)
        {
            if (g[octant] == 0) { g[octant] = 1; } // cancel node merge
        }
    }

    if (leaves[nodeIdx] != key) // mandatory key is not present
    {
        int keyPos = lastNzPlace(key);

        // only add 1 level, otherwise nodes can be added in a non-peer area,
        // exceeding the resolution of the global tree, which will result in a failure to compute
        // exact particle counts for those nodes
        constexpr int maxAddLevels = 1;
        int levelDiff              = keyPos - level;
        if (levelDiff > maxAddLevels) { status = ResolutionStatus::failed; }
        else { status = ResolutionStatus::rebalance; }

        levelDiff        = stl::min(levelDiff, maxAddLevels);
        nodeOps[nodeIdx] = stl::max(nodeOps[nodeIdx], 1 << (3 * levelDiff));
    }

    return status;
}

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
    assert(nNodes(treeLeaves) == nodeOps.size());
    ResolutionStatus status = ResolutionStatus::converged;

    for (KeyType key : mandatoryKeys)
    {
        status = std::max(enforceKeySingle(treeLeaves.data(), nodeOps.data(), nNodes(treeLeaves), key), status);
    }
    return status;
}

/*! @brief inject specified keys into a cornerstone leaf tree
 *
 * @tparam KeyType    32- or 64-bit unsigned integer
 * @param[inout] tree   cornerstone octree
 * @param[in]    keys   list of SFC keys to insert
 *
 * This function needs to insert more than just @p keys, due the cornerstone
 * invariant of consecutive nodes always having a power-of-8 difference.
 * This means that each subdividing a node, all 8 children always have to be added.
 */
template<class KeyType, class Alloc>
void injectKeys(std::vector<KeyType, Alloc>& tree, gsl::span<const KeyType> keys)
{
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

} // namespace cstone
