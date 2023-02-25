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
#include "cstone/tree/octree.hpp"
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
    KeyType nodeKey            = nodeKeys[nodeIdx];
    unsigned level             = decodePrefixLength(nodeKeys[nodeIdx]) / 3;

    if (nodeIdx)
    {
        bool countMerge = counts[parent] <= bucketSize;
        bool macMerge   = macs[parent] == 0;

        KeyType firstGroupKey = decodePlaceholderBit(nodeKeys[parent]);
        KeyType lastGroupKey  = firstGroupKey + 8 * nodeRange<KeyType>(level);
        // inFringe: nodeIdx not in focus, but at least one sibling is in the focus
        // in that case we cannot remove the nodes based on a MAC criterion
        bool inFringe = overlapTwoRanges(firstGroupKey, lastGroupKey, focusStart, focusEnd);

        if (countMerge || (macMerge && !inFringe)) { return 0; } // merge
    }

    KeyType nodeStart = decodePlaceholderBit(nodeKey);
    bool isLeaf       = childOffsets[nodeIdx] == 0;
    bool inFocus      = (nodeStart >= focusStart && nodeStart < focusEnd);
    if (isLeaf && level < maxTreeLevel<KeyType>{} && counts[nodeIdx] > bucketSize && (macs[nodeIdx] || inFocus))
    {
        return 8; // split
    }

    return 1; // default: do nothing
}

/*! @brief Overrides a 0-value of nodeOps[nodeIdx] if @p nodeIdx is the left-most descendant of a non-zero ancestor
 *
 * @tparam KeyType
 * @param nodeIdx   index of a leaf node to check
 * @param prefixes  node SFC keys in placeholder-bit format
 * @param parents   parent node indices per group of 8 siblings
 * @param nodeOps   node transformation operation in {0, 1, 8}
 * @return          0 or nodeOps[closestNonZeroAncestor]
 *
 * If @p nodeIdx is the left-most descendant of the closest ancestor with a non-zero nodeOps value, return
 * the nodeOps value of the ancestor, 0 otherwise
 */
template<class KeyType>
inline HOST_DEVICE_FUN TreeNodeIndex
nzAncestorOp(TreeNodeIndex nodeIdx, const KeyType* prefixes, const TreeNodeIndex* parents, const TreeNodeIndex* nodeOps)
{
    if (nodeIdx == 0) { return nodeOps[0]; }

    TreeNodeIndex closestNonZeroAncestor = nodeIdx;
    while (nodeOps[closestNonZeroAncestor] == 0)
    {
        closestNonZeroAncestor = parents[(closestNonZeroAncestor - 1) / 8];
    }

    if (decodePlaceholderBit(prefixes[nodeIdx]) == decodePlaceholderBit(prefixes[closestNonZeroAncestor]))
    {
        return nodeOps[closestNonZeroAncestor];
    }

    return 0;
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
void rebalanceDecisionEssential(gsl::span<const KeyType> nodeKeys,
                                const TreeNodeIndex* childOffsets,
                                const TreeNodeIndex* parents,
                                const unsigned* counts,
                                const char* macs,
                                KeyType focusStart,
                                KeyType focusEnd,
                                unsigned bucketSize,
                                TreeNodeIndex* nodeOps)
{
#pragma omp parallel for
    for (TreeNodeIndex i = 0; i < nodeKeys.ssize(); ++i)
    {
        nodeOps[i] = mergeCountAndMacOp(i, nodeKeys.data(), childOffsets, parents, counts, macs, focusStart, focusEnd,
                                        bucketSize);
    }
}

template<class KeyType>
bool protectAncestors(gsl::span<const KeyType> nodeKeys, const TreeNodeIndex* parents, TreeNodeIndex* nodeOps)
{
    int numChanges = 0;
#pragma omp parallel for reduction(+ : numChanges)
    for (TreeNodeIndex i = 0; i < nodeKeys.ssize(); ++i)
    {
        int opDecision = nzAncestorOp(i, nodeKeys.data(), parents, nodeOps);

        if (opDecision != 1) { numChanges++; }
        nodeOps[i] = opDecision;
    }
    return numChanges == 0;
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
HOST_DEVICE_FUN ResolutionStatus enforceKeySingle(KeyType key,
                                                  const KeyType* nodeKeys,
                                                  const TreeNodeIndex* childOffsets,
                                                  const TreeNodeIndex* parents,
                                                  TreeNodeIndex* nodeOps)
{
    if (key == 0 || key == nodeRange<KeyType>(0)) { return ResolutionStatus::converged; }

    ResolutionStatus status = ResolutionStatus::converged;

    KeyType nodeKeyWant    = makePrefix(key);
    TreeNodeIndex nodeIdx  = containingNode(nodeKeyWant, nodeKeys, childOffsets);
    KeyType nodeKeyHave    = nodeKeys[nodeIdx];
    unsigned nodeLevelHave = decodePrefixLength(nodeKeyHave) / 3;

    // need to undo merges of all supporting ancestors if the closest tree node would be merged
    // or the mandatory key is not there
    bool trySplit   = nodeKeyHave != nodeKeyWant && nodeLevelHave < maxTreeLevel<KeyType>{};
    bool undoMerges = nodeOps[nodeIdx] == 0 || trySplit;
    if (undoMerges && nodeIdx > 0)
    {
        status = ResolutionStatus::cancelMerge;
        // make sure no supporting ancestor of nodeIdx will be merged
        TreeNodeIndex parent = nodeIdx;
        do
        {
            parent                     = parents[(parent - 1) / 8];
            TreeNodeIndex firstSibling = childOffsets[parent];
            for (TreeNodeIndex i = firstSibling; i < firstSibling + eightSiblings; ++i)
            {
                if (nodeOps[i] == 0) { nodeOps[i] = 1; }
            }
        } while (parent != 0);
    }

    if (trySplit)
    {
        int keyPos = lastNzPlace(key);

        // only add 1 level, otherwise nodes can be added in a non-peer area,
        // exceeding the resolution of the global tree, which will result in a failure to compute
        // exact particle counts for those nodes
        constexpr int maxAddLevels = 1;
        int levelDiff              = keyPos - nodeLevelHave;
        if (levelDiff > maxAddLevels) { status = ResolutionStatus::failed; }
        else { status = ResolutionStatus::rebalance; }

        levelDiff        = stl::min(levelDiff, maxAddLevels);
        nodeOps[nodeIdx] = stl::max(nodeOps[nodeIdx], 1 << (3 * levelDiff));
    }

    return status;
}

template<class KeyType>
ResolutionStatus enforceKeys(gsl::span<const KeyType> mandatoryKeys,
                             const KeyType* nodeKeys,
                             const TreeNodeIndex* childOffsets,
                             const TreeNodeIndex* parents,
                             TreeNodeIndex* nodeOps)
{
    ResolutionStatus status = ResolutionStatus::converged;

    for (KeyType key : mandatoryKeys)
    {
        status = std::max(enforceKeySingle(key, nodeKeys, childOffsets, parents, nodeOps), status);
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
