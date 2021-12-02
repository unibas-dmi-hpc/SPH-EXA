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
 * @brief  Compute the internal part of a cornerstone octree
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * General algorithm:
 *      cornerstone octree (leaves) -> internal binary radix tree -> internal octree
 *
 * Like the cornerstone octree, the internal octree is stored in a linear memory layout
 * with tree nodes placed next to each other in a single buffer. Construction
 * is fully parallel and non-recursive and non-iterative. Traversal is possible non-recursively
 * in an iterative fashion with a local stack.
 */

#pragma once

#include <atomic>
#include <iterator>
#include <vector>

#include "cstone/cuda/annotation.hpp"
#include "cstone/primitives/gather.hpp"
#include "cstone/sfc/sfc.hpp"
#include "cstone/util/gsl-lite.hpp"

#include "btree.hpp"
#include "octree.hpp"


namespace cstone
{

/*! @brief construct internal octree from given leaves
 *
 * @tparam KeyType
 * @param[in]  leaves           cornerstone leaf cell array, length @p numLeafNodes + 1
 * @param[in]  numLeafNodes
 * @param[in]  binaryTree       binary radix tree node array, length @p numLeafNodes - 1
 * @param[out] prefixes         output octree SFC node kyes, length = numLeafNodes + (numLeafNodes-1)/7
 * @param[out] levels           output octree levels, length same as @p prefixes
 * @param[out] childOffsets     child index per node, length same as @p prefixes
 * @param[out] parents          parent index per group of 8 nodes, length is (len(prefixes) - 1) / 8
 *                              the parent index of node i is stored at parents[(i-1)/8]
 * @param[out] levelOffsets     location of first node of each level, length maxTreeLevel<KeyType>{} + 2
 *                              +1 to store the root and another +1 to store the total number of nodes
 *                              in the last element. levelOffsets[0] = 0 is the root (zero) offset,
 *                              levelOffsets[i] is the index of the first level-i node.
 */
template<class KeyType>
void constructOctree(const KeyType* leaves, TreeNodeIndex numLeafNodes, const BinaryNode<KeyType>* binaryTree,
                     KeyType* prefixes, unsigned* levels, TreeNodeIndex* childOffsets, TreeNodeIndex* parents,
                     TreeNodeIndex* levelOffsets, TreeNodeIndex* nodeOrder)
{
    // we ignore the last binary tree node which is a duplicate root node
    TreeNodeIndex nBinaryNodes = numLeafNodes - 1;

    // one extra element to store the total sum of the exclusive scan
    std::vector<TreeNodeIndex> binaryToOct(nBinaryNodes + 1);
    #pragma omp parallel for schedule(static)
    for (TreeNodeIndex i = 0; i < nBinaryNodes; ++i)
    {
        int  prefixLength = decodePrefixLength(binaryTree[i].prefix);
        bool divisibleBy3 = prefixLength % 3 == 0;
        binaryToOct[i] = (divisibleBy3) ? 1 : 0;
    }

    // stream compaction: scan and scatter
    exclusiveScan(binaryToOct.data(), binaryToOct.size());

    // nInternalOctreeNodes is also equal to prefixes[nBinaryNodes]
    TreeNodeIndex numInternalNodes = (numLeafNodes - 1) / 7;
    std::vector<TreeNodeIndex> octToBinary(numInternalNodes);

    // compaction step, scatterMap -> compacted list of binary nodes that correspond to octree nodes
    #pragma omp parallel for schedule(static)
    for (TreeNodeIndex i = 0; i < nBinaryNodes; ++i)
    {
        bool isOctreeNode = (binaryToOct[i+1] - binaryToOct[i]) == 1;
        if (isOctreeNode)
        {
            int octreeNodeIndex = binaryToOct[i];
            octToBinary[octreeNodeIndex] = i;
        }
    }

    TreeNodeIndex numNodes = numInternalNodes + numLeafNodes;

    //std::vector<TreeNodeIndex> nodeOrder(numNodes);
    std::iota(nodeOrder, nodeOrder + numNodes, 0);

    #pragma omp parallel for schedule(static)
    for (TreeNodeIndex i = 0; i < numInternalNodes; ++i)
    {
        TreeNodeIndex binaryIndex = octToBinary[i];
        levels[i]   = decodePrefixLength(binaryTree[binaryIndex].prefix) / 3;
        prefixes[i] = decodePlaceholderBit(binaryTree[binaryIndex].prefix);
    }

    std::copy(leaves, leaves + numLeafNodes, prefixes + numInternalNodes);

    #pragma omp parallel for schedule(static)
    for (TreeNodeIndex i = 0; i < numLeafNodes; ++i)
    {
        unsigned level = treeLevel(leaves[i+1] - leaves[i]);
        levels[i + numInternalNodes] = level;
    }

    /*! prefix and levels now in unsorted layout A
     *
     * binaryTree |---------------------------------------------------|
     *                    ^                      |
     *   octToBinary   |--|  |-------------------|  binaryToOct
     *                 |     V
     * prefixes   |------------|--------------------------------|
     * levels     |------------|--------------------------------|
     *              internal        leaves
     */

    sort_by_key(levels, levels + numNodes, nodeOrder);
    {
        std::vector<KeyType> tmpPrefixes(prefixes, prefixes + numNodes);
        reorder(gsl::span<const TreeNodeIndex>(nodeOrder, numNodes), tmpPrefixes.data(), prefixes, 0, numNodes);
    }

    for (unsigned level = 0; level < maxTreeLevel<KeyType>{} + 1; ++level)
    {
        auto it1 = std::lower_bound(levels, levels + numNodes, level);
        auto it2 = std::upper_bound(levels, levels + numNodes, level);
        levelOffsets[level] = TreeNodeIndex(it2 - it1);
    }

    exclusiveScan(levelOffsets, maxTreeLevel<KeyType>{} + 2);

    #pragma omp parallel for schedule(dynamic)
    for (unsigned level = 0; level < maxTreeLevel<KeyType>{} + 1; ++level)
    {
        TreeNodeIndex lvlStart = levelOffsets[level];
        sort_by_key(prefixes + lvlStart, prefixes + levelOffsets[level + 1], nodeOrder + lvlStart);
    }

    /*! prefix and levels now in sorted layout B
     *
     *  -levels is sorted in ascending order
     *  -prefix is first sorted by level, then by ascending key
     *  -nodeOrder goes from layout B to layout A (nodeOrder[i] is i's location in A)
     */

    std::vector<TreeNodeIndex> inverseNodeOrder(numNodes);
    std::iota(begin(inverseNodeOrder), end(inverseNodeOrder), 0);

    // temporarily repurpose childOffsets as sort key
    std::copy(nodeOrder, nodeOrder + numNodes, childOffsets);
    sort_by_key(childOffsets, childOffsets + numNodes, begin(inverseNodeOrder));
    std::fill(childOffsets, childOffsets + numNodes, 0);

    // loop over octree nodes index in layout A
    #pragma omp parallel for schedule(static)
    for (TreeNodeIndex idxA = 0; idxA < numInternalNodes; ++idxA)
    {
        TreeNodeIndex binaryIndex = octToBinary[idxA];
        TreeNodeIndex firstChild  = binaryTree[binaryTree[binaryTree[binaryIndex].child[0]].child[0]].child[0];

        // octree node index in layout B
        TreeNodeIndex idxB = inverseNodeOrder[idxA];

        // child node index in layout A
        TreeNodeIndex childA =
            (isLeafIndex(firstChild)) ? loadLeafIndex(firstChild) + numInternalNodes : binaryToOct[firstChild];

        // node index in layout B
        TreeNodeIndex childB = inverseNodeOrder[childA];

        childOffsets[idxB]        = childB;
        parents[(childB - 1) / 8] = idxB;
    }
}

template<class KeyType>
class TdOctree
{
public:
    /*! @brief default ctor
     *
     * Apart from the root at level 0, there are maxTreeLevel<KeyType>{} non-trivial levels.
     * For convenience, we also store the root offset, even though the root is always a single node
     * at offset 0. So in total there are maxTreeLevel<KeyType>{}+1 levels and we need
     * another +1 to store the last upper bound which is equal to the total number of nodes in the tree.
     */
    TdOctree()
        : levelOffsets_(maxTreeLevel<KeyType>{} + 2)
    {
    }

    void update(const KeyType* firstLeaf, TreeNodeIndex newNumLeafNodes)
    {
        updateInternalTree(firstLeaf, newNumLeafNodes);
    }

    //! @brief total number of nodes in the tree
    [[nodiscard]] inline TreeNodeIndex numTreeNodes() const
    {
        return levelOffsets_.back();
    }

    [[nodiscard]] inline TreeNodeIndex numTreeNodes(unsigned level) const
    {
        assert(level <= maxTreeLevel<KeyType>{});
        return levelOffsets_[level + 1] - levelOffsets_[level];
    }

    //! @brief number of leaf nodes in the tree
    [[nodiscard]] inline TreeNodeIndex numLeafNodes() const
    {
        return numLeafNodes_;
    }

    //! @brief number of internal nodes in the tree, equal to (numLeafNodes()-1) / 7
    [[nodiscard]] inline TreeNodeIndex numInternalNodes() const
    {
        return numInternalNodes_;
    }

    /*! @brief check whether node is a leaf
     *
     * @param[in] node    node index, range [0:numTreeNodes()]
     * @return            true or false
     */
    [[nodiscard]] inline bool isLeaf(TreeNodeIndex node) const
    {
        assert(node < numTreeNodes());
        return childOffsets_[node] == 0;
    }

    //! @brief check if node is the root node
    [[nodiscard]] inline bool isRoot(TreeNodeIndex node) const
    {
        return node == 0;
    }

    /*! @brief return child node index
     *
     * @param[in] node    node index, range [0:numInternalNodes()]
     * @param[in] octant  octant index, range [0:8]
     * @return            child node index, range [0:nNodes()]
     *
     * If @p node is not internal, behavior is undefined.
     * Query with isLeaf(node) before calling this function.
     */
    [[nodiscard]] inline TreeNodeIndex child(TreeNodeIndex node, int octant) const
    {
        assert(node < TreeNodeIndex(childOffsets_.size()));

        return childOffsets_[node] + octant;
    }

    /*! @brief index of parent node
     *
     * Note: the root node is its own parent
     */
    [[nodiscard]] inline TreeNodeIndex parent(TreeNodeIndex node) const
    {
        return parents_[(node -1) / 8];
    }

    //! @brief lowest SFC key contained int the geometrical box of @p node
    [[nodiscard]] inline KeyType codeStart(TreeNodeIndex node) const
    {
        return prefixes_[node];
    }

    //! @brief highest SFC key contained in the geometrical box of @p node
    [[nodiscard]] inline KeyType codeEnd(TreeNodeIndex node) const
    {
        return prefixes_[node] + nodeRange<KeyType>(levels_[node]);
    }

    /*! @brief octree subdivision level for @p node
     *
     * Returns 0 for the root node. Highest value is maxTreeLevel<KeyType>{}.
     */
    [[nodiscard]] inline unsigned level(TreeNodeIndex node) const
    {
        return levels_[node];
    }

    [[nodiscard]] inline TreeNodeIndex levelOffset(unsigned level) const
    {
        return levelOffsets_[level];
    }

    /*! @brief returns index of @p node in the cornerstone tree used for construction
     *
     * @param node  node in [0:numTreeNodes()]
     * @return      the index in the cornerstone tree used for construction if @p node is a leaf
     *              such that this->codeStart(node) == cstoneTree[this->cstoneIndex(node)]
     *              If node is not a leaf, the return value is negative.
     */
    [[nodiscard]] inline TreeNodeIndex cstoneIndex(TreeNodeIndex node) const
    {
        return nodeOrder_[node] - numInternalNodes_;
    }

    /*! @brief finds the index of the node with SFC key range [startKey:endKey]
     *
     * @param startKey   lower SFC key
     * @param endKey     upper SFC key
     * @return           The index i of the node that satisfies codeStart(i) == startKey
     *                   and codeEnd(i) == endKey, or numTreeNodes() if no such node exists.
     */
    TreeNodeIndex locate(KeyType startKey, KeyType endKey) const
    {
        unsigned level = treeLevel(endKey - startKey);

        return std::lower_bound(prefixes_.begin() + levelOffsets_[level],
                                prefixes_.begin() + levelOffsets_[level + 1], startKey) -
               prefixes_.begin();
    }

private:

    //! @brief regenerates the internal tree based on (a changed) cstoneTree_
    void updateInternalTree(const KeyType* leaves, TreeNodeIndex numLeafNodes)
    {
        numLeafNodes_          = numLeafNodes;
        numInternalNodes_      = (numLeafNodes - 1) / 7;
        TreeNodeIndex numNodes = numInternalNodes_ + numLeafNodes_;

        std::vector<BinaryNode<KeyType>> binaryTree(numLeafNodes);
        createBinaryTree(leaves, numLeafNodes, binaryTree.data());

        prefixes_.resize(numNodes);
        levels_.resize(numNodes);
        childOffsets_.resize(numNodes);
        nodeOrder_.resize(numNodes);

        parents_.resize((numNodes - 1) / 8);

        constructOctree(leaves, numLeafNodes, binaryTree.data(), prefixes_.data(), levels_.data(), childOffsets_.data(),
                        parents_.data(), levelOffsets_.data(), nodeOrder_.data());
    }

    TreeNodeIndex numLeafNodes_{0};
    TreeNodeIndex numInternalNodes_{0};

    std::vector<KeyType> prefixes_;

    //! @brief subdivision level of each node
    std::vector<unsigned> levels_;

    //! @brief node offsets of first child for each node
    std::vector<TreeNodeIndex> childOffsets_;

    std::vector<TreeNodeIndex> parents_;

    //! @brief stores index of first node for each level
    std::vector<TreeNodeIndex> levelOffsets_;

    //! @brief stores node locations in the cornerstone leaf array
    std::vector<TreeNodeIndex> nodeOrder_;
};

template<class T, class KeyType, class CombinationFunction>
void upsweep(const TdOctree<KeyType>& octree, T* quantities, CombinationFunction combinationFunction)
{
    unsigned currentLevel = octree.level(octree.numTreeNodes() - 1);

    for ( ; currentLevel != 0; --currentLevel)
    {
        TreeNodeIndex start = octree.levelOffset(currentLevel);
        TreeNodeIndex end   = octree.levelOffset(currentLevel + 1);
        for (TreeNodeIndex i = start; i < end; ++i)
        {
            if (!octree.isLeaf(i))
            {
                 quantities[i] = combinationFunction(quantities[octree.child(i, 0)],
                                                     quantities[octree.child(i, 1)],
                                                     quantities[octree.child(i, 2)],
                                                     quantities[octree.child(i, 3)],
                                                     quantities[octree.child(i, 4)],
                                                     quantities[octree.child(i, 5)],
                                                     quantities[octree.child(i, 6)],
                                                     quantities[octree.child(i, 7)]);
            }
        }
    }
}

} // namespace cstone
