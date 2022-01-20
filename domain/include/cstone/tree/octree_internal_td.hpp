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

/*! @brief determine which binary nodes correspond to octree nodes
 *
 * @tparam     KeyType     unsigned 32- or 64-bit integer
 * @param[in]  binaryTree  binary radix tree nodes, length @p numNodes
 * @param[in]  numNodes    number of binary radix tree nodes
 * @param[out] binaryToOct for each binary node, store 1 if prefix bit length is divisible by 3
 */
template<class KeyType>
void enumeratePrefixes(const BinaryNode<KeyType>* binaryTree, TreeNodeIndex numNodes, TreeNodeIndex* binaryToOct)
{
    #pragma omp parallel for schedule(static)
    for (TreeNodeIndex tid = 0; tid < numNodes; ++tid)
    {
        int  prefixLength = decodePrefixLength(binaryTree[tid].prefix);
        bool divisibleBy3 = prefixLength % 3 == 0;
        binaryToOct[tid] = (divisibleBy3) ? 1 : 0;
    }
}

/*! @brief map octree nodes back to binary nodes
 *
 * @param[in]  binaryToOct     translation map from binary to octree nodes, length @p numBinaryNodes
 * @param[in]  numBinaryNodes  number of binary tree nodes
 * @param[out] octToBinary     the inversion of binaryToOct, octToBinary[binaryToOct[i]] == i
 */
inline void translateToOct(const TreeNodeIndex* binaryToOct, int numBinaryNodes, TreeNodeIndex* octToBinary)
{
    #pragma omp parallel for schedule(static)
    for (TreeNodeIndex tid = 0; tid < numBinaryNodes; ++tid)
    {
        bool isOctreeNode = (binaryToOct[tid + 1] - binaryToOct[tid]) == 1;
        if (isOctreeNode)
        {
            int octreeNodeIndex          = binaryToOct[tid];
            octToBinary[octreeNodeIndex] = tid;
        }
    }
}

/*! @brief combine internal and leaf tree parts into single arrays in the binary tree order
 *
 * @tparam KeyType               unsigned 32- or 64-bit integer
 * @param[in]  binaryTree        binary radix tree nodes, length numNodes - numInternalNodes - 1
 * @param[in]  numInternalNodes  number of internal (output) octree nodes
 * @param[in]  numNodes          total number of nodes
 * @param[in]  leaves            cornerstone SFC keys used to compute binaryTree, length numNodes - numInternalNodes + 1
 * @param[in]  octToBinary       translation map from octree to binary node indices
 * @param[out] prefixes          output octree SFC keys, length @p numNodes
 *                               NOTE: keys are prefixed with Warren-Salmon placeholder bits!
 * @param[out] nodeOrder         iota 0,1,2,3,... sequence for later use, length @p numNodes
 * @param[out] inverseNodeOrder  iota 0,1,2,3,... sequence for later use, length @p numNodes
 *
 * Unsorted binary radix tree ordering: first all internal nodes, then leaf nodes
 *
 *    binaryTree |---------------------------------------------------|
 *                       ^                      |
 *      octToBinary   |--|  |-------------------|  binaryToOct
 *                    |     V
 *    prefixes   |------------|--------------------------------|
 *    levels     |------------|--------------------------------|
 *                 internal        leaves
 */
template<class KeyType>
void createUnsortedLayout(const BinaryNode<KeyType>* binaryTree,
                          TreeNodeIndex numInternalNodes,
                          TreeNodeIndex numNodes,
                          const KeyType* leaves,
                          const TreeNodeIndex* octToBinary,
                          KeyType* prefixes,
                          TreeNodeIndex* nodeOrder,
                          TreeNodeIndex* inverseNodeOrder)
{
    #pragma omp parallel for schedule(static)
    for (TreeNodeIndex tid = 0; tid < numInternalNodes; ++tid)
    {
        TreeNodeIndex binaryIndex = octToBinary[tid];
        prefixes[tid]             = binaryTree[binaryIndex].prefix;

        nodeOrder[tid]        = tid;
        inverseNodeOrder[tid] = tid;
    }
    #pragma omp parallel for schedule(static)
    for (TreeNodeIndex tid = numInternalNodes; tid < numNodes; ++tid)
    {
        TreeNodeIndex leafIdx = tid - numInternalNodes;
        unsigned level        = treeLevel(leaves[leafIdx + 1] - leaves[leafIdx]);
        prefixes[tid]         = encodePlaceholderBit(leaves[leafIdx], 3 * level);

        nodeOrder[tid]        = tid;
        inverseNodeOrder[tid] = tid;
    }
}

/*! @brief extract parent/child relationships from binary tree and translate to sorted order
 *
 * @tparam     KeyType           unsigned 32- or 64-bit integer
 * @param[in]  binaryTree        binary radix tree nodes
 * @param[in]  numInternalNodes  number of internal octree nodes
 * @param[in]  octToBinary       internal octree to binary node index translation map, length @p numInternalNodes
 * @param[in]  binaryToOct       binary node to internal octree node index translation map
 * @param[in]  inverseNodeOrder  translation map from unsorted layout to level/SFC sorted octree layout
 *                               length is total number of octree nodes, internal + leaves
 * @param[out] childOffsets      octree node index of first child for each node, length is total number of nodes
 * @param[out] parents           parent index of for each node which is the first of 8 siblings
 *                               i.e. the parent of node i is stored at parents[(i - 1)/8]
 */
template<class KeyType>
void linkTree(const BinaryNode<KeyType>* binaryTree,
              TreeNodeIndex numInternalNodes,
              const TreeNodeIndex* octToBinary,
              const TreeNodeIndex* binaryToOct,
              const TreeNodeIndex* inverseNodeOrder,
              TreeNodeIndex* childOffsets,
              TreeNodeIndex* parents)
{
    // loop over octree nodes index in unsorted layout A
    #pragma omp parallel for schedule(static)
    for (TreeNodeIndex idxA = 0; idxA < numInternalNodes; ++idxA)
    {
        TreeNodeIndex binaryIndex = octToBinary[idxA];
        TreeNodeIndex firstChild  = binaryTree[binaryTree[binaryTree[binaryIndex].child[0]].child[0]].child[0];

        // octree node index in sorted layout B
        TreeNodeIndex idxB = inverseNodeOrder[idxA];

        // child node index in unsorted layout A
        TreeNodeIndex childA =
            (isLeafIndex(firstChild)) ? loadLeafIndex(firstChild) + numInternalNodes : binaryToOct[firstChild];

        // node index in layout B
        TreeNodeIndex childB = inverseNodeOrder[childA];
        // an internal node must have a valid child
        assert(childB > 0);

        childOffsets[idxB] = childB;
        // We only store the parent once for every group of 8 siblings. This works as long as each node always has
        // 8 siblings. Subtract one because the root has no siblings.
        parents[(childB - 1) / 8] = idxB;
    }
}

//! @brief determine the octree subdivision level boundaries
template<class KeyType>
void getLevelRange(const KeyType* nodeKeys, TreeNodeIndex numNodes, TreeNodeIndex* levelRange)
{
    levelRange[0] = 0;
    levelRange[1] = 1;
    for (unsigned level = 2; level <= maxTreeLevel<KeyType>{}; ++level)
    {
        auto it = std::lower_bound(nodeKeys, nodeKeys + numNodes, 3 * level,
                                   [](KeyType k, KeyType val) { return decodePrefixLength(k) < val; });
        levelRange[level] = TreeNodeIndex(it - nodeKeys);
    }
    levelRange[maxTreeLevel<KeyType>{} + 1] = numNodes;
}

/*! @brief functor to sort octree nodes first according to level, then by SFC key
 *
 * Note: takes SFC keys with Warren-Salmon placeholder bits in place as arguments
 */
template<class KeyType>
struct compareLevelThenPrefixCpu
{
    HOST_DEVICE_FUN bool operator()(KeyType a, KeyType b) const
    {
        return util::tuple<unsigned, KeyType>{decodePrefixLength(a), a} <
               util::tuple<unsigned, KeyType>{decodePrefixLength(b), b};
    }
};

/*! @brief construct the internal octree part of a given octree leaf cell array on the GPU
 *
 * @tparam       KeyType     unsigned 32- or 64-bit integer
 * @param[in]    cstoneTree  GPU buffer with the SFC leaf cell keys
 */
template<class KeyType>
void buildInternalOctreeGpu(const KeyType* cstoneTree,
                            TreeNodeIndex numLeafNodes,
                            TreeNodeIndex numInternalNodes,
                            KeyType* prefixes,
                            TreeNodeIndex* childOffsets,
                            TreeNodeIndex* parents,
                            TreeNodeIndex* levelRange,
                            TreeNodeIndex* nodeOrder,
                            TreeNodeIndex* inverseNodeOrder,
                            BinaryNode<KeyType>* binaryTree,
                            TreeNodeIndex* binaryToOct,
                            TreeNodeIndex* octToBinary)
{
    createBinaryTree(cstoneTree, numLeafNodes, binaryTree);

    // we ignore the last binary tree node which is a duplicate root node
    TreeNodeIndex numBinaryNodes = numLeafNodes - 1;
    enumeratePrefixes(binaryTree, numBinaryNodes, binaryToOct);
    exclusiveScan(binaryToOct, numLeafNodes);
    translateToOct(binaryToOct, numBinaryNodes, octToBinary);
    TreeNodeIndex numNodes = numInternalNodes + numLeafNodes;
    createUnsortedLayout(binaryTree, numInternalNodes, numNodes, cstoneTree, octToBinary, prefixes, nodeOrder,
                         inverseNodeOrder);
    sort_by_key(prefixes, prefixes + numNodes, nodeOrder, compareLevelThenPrefixCpu<KeyType>{});
    // arrays now in sorted layout B

    // temporarily repurpose childOffsets as space for sort key
    std::copy(nodeOrder, nodeOrder + numNodes, childOffsets);
    sort_by_key(childOffsets, childOffsets + numNodes, inverseNodeOrder);
    std::fill(childOffsets, childOffsets + numNodes, 0);

    linkTree(binaryTree, numInternalNodes, octToBinary, binaryToOct, inverseNodeOrder, childOffsets, parents);
    getLevelRange(prefixes, numNodes, levelRange);
}

template<class KeyType>
class TdOctree
{
public:
    TdOctree() = default;

    void update(const KeyType* firstLeaf, TreeNodeIndex newNumLeafNodes)
    {
        updateInternalTree(firstLeaf, newNumLeafNodes);
    }

    //! @brief total number of nodes in the tree
    [[nodiscard]] inline TreeNodeIndex numTreeNodes() const
    {
        return levelRange_.back();
    }

    [[nodiscard]] inline TreeNodeIndex numTreeNodes(unsigned level) const
    {
        assert(level <= maxTreeLevel<KeyType>{});
        return levelRange_[level + 1] - levelRange_[level];
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
    [[nodiscard]] inline TreeNodeIndex parent(TreeNodeIndex node) const { return parents_[(node - 1) / 8]; }

    //! @brief lowest SFC key contained int the geometrical box of @p node
    [[nodiscard]] inline KeyType codeStart(TreeNodeIndex node) const
    {
        return decodePlaceholderBit(prefixes_[node]);
    }

    //! @brief highest SFC key contained in the geometrical box of @p node
    [[nodiscard]] inline KeyType codeEnd(TreeNodeIndex node) const
    {
        KeyType prefix = prefixes_[node];
        assert(decodePrefixLength(prefix) % 3  == 0);
        return decodePlaceholderBit(prefix) + (1ul << (3 * maxTreeLevel<KeyType>{} - decodePrefixLength(prefix)));
    }

    /*! @brief octree subdivision level for @p node
     *
     * Returns 0 for the root node. Highest value is maxTreeLevel<KeyType>{}.
     */
    [[nodiscard]] inline unsigned level(TreeNodeIndex node) const
    {
        return decodePrefixLength(prefixes_[node]) / 3;
    }

    //! @brief return the index of the first node at subdivision level @p level
    [[nodiscard]] inline TreeNodeIndex levelOffset(unsigned level) const
    {
        return levelRange_[level];
    }

    /*! @brief convert a leaf index (indexed from first leaf starting from 0) to 0-indexed from root
     *
     * @param[in] node    leaf node index, range [0:numLeafNodes()]
     * @return            octree index, relative to the root node
     */
    [[nodiscard]] inline TreeNodeIndex toInternal(TreeNodeIndex node) const
    {
        assert(size_t(node + numInternalNodes()) < inverseNodeOrder_.size());
        return inverseNodeOrder_[node + numInternalNodes()];
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
        assert(size_t(node) < nodeOrder_.size());
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

        return std::lower_bound(prefixes_.begin() + levelRange_[level], prefixes_.begin() + levelRange_[level + 1],
                                startKey, [](KeyType k, KeyType val) { return decodePlaceholderBit(k) < val; }) -
               prefixes_.begin();
    }

private:
    //! @brief regenerates the internal tree based on (a changed) cstoneTree_
    void updateInternalTree(const KeyType* leaves, TreeNodeIndex numLeafNodes)
    {
        resize(numLeafNodes);
        buildInternalOctreeGpu(leaves, numLeafNodes_, numInternalNodes_, prefixes_.data(), childOffsets_.data(),
                               parents_.data(), levelRange_.data(), nodeOrder_.data(), inverseNodeOrder_.data(),
                               binaryTree_.data(), binaryToOct_.data(), octToBinary_.data());
    }

    void resize(TreeNodeIndex numCsLeafNodes)
    {
        numLeafNodes_          = numCsLeafNodes;
        numInternalNodes_      = (numLeafNodes_ - 1) / 7;
        TreeNodeIndex numNodes = numLeafNodes_ + numInternalNodes_;

        prefixes_.resize(numNodes);
        childOffsets_.resize(numNodes);
        parents_.resize((numNodes - 1) / 8);
        /*! Apart from the root at level 0, there are maxTreeLevel<KeyType>{} non-trivial levels.
         * For convenience, we also store the root offset, even though the root is always a single node
         * at offset 0. So in total there are maxTreeLevel<KeyType>{}+1 levels and we need
         * another +1 to store the last upper bound which is equal to the total number of nodes in the tree.
         */
        levelRange_.resize(maxTreeLevel<KeyType>{} + 2);

        nodeOrder_.resize(numNodes);
        inverseNodeOrder_.resize(numNodes);

        binaryTree_.resize(numLeafNodes_);
        binaryToOct_.resize(numLeafNodes_);
        octToBinary_.resize(numInternalNodes_);
    }

    TreeNodeIndex numLeafNodes_{0};
    TreeNodeIndex numInternalNodes_{0};

    //! @brief the SFC key and level of each node (Warren-Salmon placeholder-bit), length = numNodes
    std::vector<KeyType> prefixes_;
    //! @brief the index of the first child of each node, a value of 0 indicates a leaf, length = numNodes
    std::vector<TreeNodeIndex> childOffsets_;
    //! @brief stores the parent index for every group of 8 sibling nodes, length the (numNodes - 1) / 8
    std::vector<TreeNodeIndex> parents_;
    //! @brief store the first node index of every tree level, length = maxTreeLevel + 2
    std::vector<TreeNodeIndex> levelRange_;

    //! @brief maps between the (level-key) sorted layout B and the unsorted intermediate binary layout A
    std::vector<TreeNodeIndex> nodeOrder_;
    std::vector<TreeNodeIndex> inverseNodeOrder_;

    //! @brief temporary storage for binary tree nodes used during construction
    std::vector<BinaryNode<KeyType>> binaryTree_;
    //! @brief temporary index maps between the binary tree and octree used during construction
    std::vector<TreeNodeIndex> binaryToOct_;
    std::vector<TreeNodeIndex> octToBinary_;
};

template<class T, class KeyType, class CombinationFunction>
void upsweep(const TdOctree<KeyType>& octree, T* quantities, CombinationFunction combinationFunction)
{
    int currentLevel = octree.level(octree.numTreeNodes() - 1);

    for ( ; currentLevel >= 0; --currentLevel)
    {
        TreeNodeIndex start = octree.levelOffset(currentLevel);
        TreeNodeIndex end   = octree.levelOffset(currentLevel + 1);
        #pragma omp parallel for schedule(static)
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
