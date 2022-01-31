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

#include <iterator>
#include <vector>

#include "cstone/cuda/annotation.hpp"
#include "cstone/primitives/gather.hpp"
#include "cstone/sfc/sfc.hpp"
#include "cstone/tree/octree.hpp"
#include "cstone/util/gsl-lite.hpp"

namespace cstone
{

/*! @brief map a binary node index to an octree node index
 *
 * @tparam KeyType    32- or 64-bit unsigned integer
 * @param  key        a cornerstone leaf cell key
 * @param  level      the subdivision level of @p key
 * @return            the index offset
 *
 * if
 *      - cstree is a cornerstone leaf array
 *      - l = commonPrefix(cstree[j], cstree[j+1]), l % 3 == 0
 *      - k = cstree[j]
 *
 * then i = (j + binaryKeyWeight(k, l) / 7 equals the index of the internal octree node with key k,
 * see unit test of this function for an illustration
 */
template<class KeyType>
HOST_DEVICE_FUN constexpr TreeNodeIndex binaryKeyWeight(KeyType key, unsigned level)
{
    TreeNodeIndex ret = 0;
    for (unsigned l = 1; l <= level + 1; ++l)
    {
        unsigned digit = octalDigit(key, l);
        ret += digitWeight(digit);
    }
    return ret;
}

/*! @brief combine internal and leaf tree parts into a single array with the nodeKey prefixes
 *
 * @tparam     KeyType           unsigned 32- or 64-bit integer
 * @param[in]  leaves            cornerstone SFC keys, length numLeafNodes + 1
 * @param[in]  numInternalNodes  number of internal octree nodes
 * @param[in]  numLeafNodes      total number of nodes
 * @param[in]  binaryToOct       translation map from binary to octree nodes
 * @param[out] prefixes          output octree SFC keys, length @p numInternalNodes + numLeafNodes
 *                               NOTE: keys are prefixed with Warren-Salmon placeholder bits!
 * @param[out] nodeOrder         iota 0,1,2,3,... sequence for later use, length same as @p prefixes
 */
template<class KeyType>
void createUnsortedLayoutCpu(const KeyType* leaves,
                             TreeNodeIndex numInternalNodes,
                             TreeNodeIndex numLeafNodes,
                             KeyType* prefixes,
                             TreeNodeIndex* nodeOrder)
{
    #pragma omp parallel for schedule(static)
    for (TreeNodeIndex tid = 0; tid < numLeafNodes; ++tid)
    {
        KeyType key                       = leaves[tid];
        unsigned level                    = treeLevel(leaves[tid + 1] - key);
        prefixes[tid + numInternalNodes]  = encodePlaceholderBit(key, 3 * level);
        nodeOrder[tid + numInternalNodes] = tid + numInternalNodes;

        unsigned prefixLength = commonPrefix(key, leaves[tid + 1]);
        if (prefixLength % 3 == 0 && tid < numLeafNodes - 1)
        {
            TreeNodeIndex octIndex = (tid + binaryKeyWeight(key, prefixLength / 3)) / 7;
            prefixes[octIndex]     = encodePlaceholderBit(key, prefixLength);
            nodeOrder[octIndex]    = octIndex;
        }
    }
}

/*! @brief extract parent/child relationships from binary tree and translate to sorted order
 *
 * @tparam     KeyType           unsigned 32- or 64-bit integer
 * @param[in]  prefixes          octree node prefixes in Warren-Salmon format
 * @param[in]  numInternalNodes  number of internal octree nodes
 * @param[in]  inverseNodeOrder  translation map from unsorted layout to level/SFC sorted octree layout
 *                               length is total number of octree nodes, internal + leaves
 * @param[in]  levelRange        indices of the first node at each level
 * @param[out] childOffsets      octree node index of first child for each node, length is total number of nodes
 * @param[out] parents           parent index of for each node which is the first of 8 siblings
 *                               i.e. the parent of node i is stored at parents[(i - 1)/8]
 */
template<class KeyType>
void linkTreeCpu(const KeyType* prefixes,
                 TreeNodeIndex numInternalNodes,
                 const TreeNodeIndex* inverseNodeOrder,
                 const TreeNodeIndex* levelRange,
                 TreeNodeIndex* childOffsets,
                 TreeNodeIndex* parents)
{
    #pragma omp parallel for schedule(static)
    for (TreeNodeIndex i = 0; i < numInternalNodes; ++i)
    {
        TreeNodeIndex idxA    = inverseNodeOrder[i];
        KeyType prefix        = prefixes[idxA];
        KeyType nodeKey       = decodePlaceholderBit(prefix);
        unsigned prefixLength = decodePrefixLength(prefix);
        unsigned level        = prefixLength / 3;
        assert(level < maxTreeLevel<KeyType>{});

        KeyType childPrefix = encodePlaceholderBit(nodeKey, prefixLength + 3);

        TreeNodeIndex leafSearchStart = levelRange[level + 1];
        TreeNodeIndex leafSearchEnd   = levelRange[level + 2];
        TreeNodeIndex childIdx =
            stl::lower_bound(prefixes + leafSearchStart, prefixes + leafSearchEnd, childPrefix) - prefixes;

        if (childIdx != leafSearchEnd && childPrefix == prefixes[childIdx])
        {
            childOffsets[idxA] = childIdx;
            // We only store the parent once for every group of 8 siblings.
            // This works as long as each node always has 8 siblings.
            // Subtract one because the root has no siblings.
            parents[(childIdx - 1) / 8] = idxA;
        }
    }
}

//! @brief determine the octree subdivision level boundaries
template<class KeyType>
void getLevelRangeCpu(const KeyType* nodeKeys, TreeNodeIndex numNodes, TreeNodeIndex* levelRange)
{
    for (unsigned level = 0; level <= maxTreeLevel<KeyType>{}; ++level)
    {
        auto it = std::lower_bound(nodeKeys, nodeKeys + numNodes, encodePlaceholderBit(KeyType(0), 3 * level));
        levelRange[level] = TreeNodeIndex(it - nodeKeys);
    }
    levelRange[maxTreeLevel<KeyType>{} + 1] = numNodes;
}

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
                            TreeNodeIndex* inverseNodeOrder)
{
    TreeNodeIndex numNodes = numInternalNodes + numLeafNodes;
    createUnsortedLayoutCpu(cstoneTree, numInternalNodes, numLeafNodes, prefixes, nodeOrder);
    sort_by_key(prefixes, prefixes + numNodes, nodeOrder);

    #pragma omp parallel for schedule(static)
    for (TreeNodeIndex i = 0; i < numNodes; ++i)
    {
        inverseNodeOrder[nodeOrder[i]] = i;
    }
    getLevelRangeCpu(prefixes, numNodes, levelRange);

    std::fill(childOffsets, childOffsets + numNodes, 0);
    linkTreeCpu(prefixes, numInternalNodes, inverseNodeOrder, levelRange, childOffsets, parents);
}

template<class K> class FocusedOctreeCore;

template<class KeyType>
class Octree
{
public:
    Octree() = default;

    //! @brief update tree, copying from externally provided leaf keys
    void update(const KeyType* leaves, TreeNodeIndex numLeafNodes)
    {
        cstoneTree_.resize(numLeafNodes + 1);
        omp_copy(leaves, leaves + numLeafNodes + 1, cstoneTree_.data());

        updateInternalTree();
    }

    //! @brief regenerates the internal tree, assuming cstoneTree_ has been changed from the outside
    void updateInternalTree()
    {
        resize(nNodes(cstoneTree_));
        buildInternalOctreeGpu(cstoneTree_.data(), numLeafNodes_, numInternalNodes_, prefixes_.data(),
                               childOffsets_.data(), parents_.data(), levelRange_.data(), nodeOrder_.data(),
                               inverseNodeOrder_.data());
    }

    //! @brief rebalance based on leaf counts only, optimized version that avoids unnecessary allocations
    bool rebalance(unsigned bucketSize, gsl::span<const unsigned> counts)
    {
        assert(counts.size() == numLeafNodes_);
        bool converged =
            rebalanceDecision(cstoneTree_.data(), counts.data(), numLeafNodes_, bucketSize, nodeOrder_.data());
        rebalanceTree(cstoneTree_, prefixes_, nodeOrder_.data());
        swap(cstoneTree_, prefixes_);

        updateInternalTree();
        return converged;
    }

    //! @brief return a const view of the cstone leaf array
    gsl::span<const KeyType> treeLeaves() const { return cstoneTree_; }
    //! @brief return const pointer to node(cell) SFC keys
    const KeyType* nodeKeys() const { return prefixes_.data(); }
    //! @brief return const pointer to child offsets array
    const TreeNodeIndex* childOffsets() const { return childOffsets_.data(); }
    //! @brief return const pointer to the cell parents array
    const TreeNodeIndex* parents() const { return parents_.data(); }

    //! @brief total number of nodes in the tree
    inline TreeNodeIndex numTreeNodes() const { return levelRange_.back(); }
    //! @brief return number of nodes per tree level
    inline TreeNodeIndex numTreeNodes(unsigned level) const { return levelRange_[level + 1] - levelRange_[level]; }
    //! @brief number of leaf nodes in the tree
    inline TreeNodeIndex numLeafNodes() const { return numLeafNodes_; }
    //! @brief number of internal nodes in the tree, equal to (numLeafNodes()-1) / 7
    inline TreeNodeIndex numInternalNodes() const { return numInternalNodes_; }

    /*! @brief check whether node is a leaf
     *
     * @param[in] node    node index, range [0:numTreeNodes()]
     * @return            true or false
     */
    inline bool isLeaf(TreeNodeIndex node) const { return childOffsets_[node] == 0; }

    //! @brief check if node is the root node
    inline bool isRoot(TreeNodeIndex node) const { return node == levelRange_[0]; }

    /*! @brief return child node index
     *
     * @param[in] node    node index, range [0:numInternalNodes()]
     * @param[in] octant  octant index, range [0:8]
     * @return            child node index, range [0:nNodes()]
     *
     * If @p node is not internal, behavior is undefined.
     * Query with isLeaf(node) before calling this function.
     */
    inline TreeNodeIndex child(TreeNodeIndex node, int octant) const
    {
        assert(node < TreeNodeIndex(childOffsets_.size()));
        return childOffsets_[node] + octant;
    }

    //! @brief Index of parent node. Note: the root node is its own parent
    inline TreeNodeIndex parent(TreeNodeIndex node) const { return node ? parents_[(node - 1) / 8] : 0; }

    //! @brief lowest SFC key contained int the geometrical box of @p node
    inline KeyType codeStart(TreeNodeIndex node) const { return decodePlaceholderBit(prefixes_[node]); }

    //! @brief highest SFC key contained in the geometrical box of @p node
    inline KeyType codeEnd(TreeNodeIndex node) const
    {
        KeyType prefix = prefixes_[node];
        assert(decodePrefixLength(prefix) % 3  == 0);
        return decodePlaceholderBit(prefix) + (1ul << (3 * maxTreeLevel<KeyType>{} - decodePrefixLength(prefix)));
    }

    /*! @brief octree subdivision level for @p node
     *
     * Returns 0 for the root node. Highest value is maxTreeLevel<KeyType>{}.
     */
    inline unsigned level(TreeNodeIndex node) const { return decodePrefixLength(prefixes_[node]) / 3; }

    //! @brief return the index of the first node at subdivision level @p level
    inline TreeNodeIndex levelOffset(unsigned level) const { return levelRange_[level]; }

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

    /*! @brief extract elements corresponding to leaf nodes and arrange in cstone (ascending SFC key) order
     *
     * @param[in]   in   input sequence of length numTreeNodes()
     * @param[out]  out  output sequence of length numLeafNodes()
     */
    template<class T>
    void extractLeaves(gsl::span<const T> in, gsl::span<T> out) const
    {
        assert(in.size() >= size_t(numTreeNodes()));
        assert(out.size() >= size_t(numLeafNodes()));

        #pragma omp parallel for schedule(static)
        for (TreeNodeIndex i = 0; i < numLeafNodes_; ++i)
        {
            out[i] = in[inverseNodeOrder_[i + numInternalNodes_]];
        }
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
    friend class FocusedOctreeCore<KeyType>;

    void resize(TreeNodeIndex numCsLeafNodes)
    {
        numLeafNodes_          = numCsLeafNodes;
        numInternalNodes_      = (numLeafNodes_ - 1) / 7;
        TreeNodeIndex numNodes = numLeafNodes_ + numInternalNodes_;

        prefixes_.resize(numNodes);
        // +1 to accommodate nodeOffsets in FocusedOctreeCore::update when numNodes == 1
        childOffsets_.resize(numNodes + 1);
        parents_.resize((numNodes - 1) / 8);
        /*! Apart from the root at level 0, there are maxTreeLevel<KeyType>{} non-trivial levels.
         * For convenience, we also store the root offset, even though the root is always a single node
         * at offset 0. So in total there are maxTreeLevel<KeyType>{}+1 levels and we need
         * another +1 to store the last upper bound which is equal to the total number of nodes in the tree.
         */
        levelRange_.resize(maxTreeLevel<KeyType>{} + 2);

        nodeOrder_.resize(numNodes);
        inverseNodeOrder_.resize(numNodes);
    }

    TreeNodeIndex numLeafNodes_{0};
    TreeNodeIndex numInternalNodes_{0};

    using Alloc = util::DefaultInitAdaptor<TreeNodeIndex>;

    //! @brief the SFC key and level of each node (Warren-Salmon placeholder-bit), length = numNodes
    std::vector<KeyType> prefixes_;
    //! @brief the index of the first child of each node, a value of 0 indicates a leaf, length = numNodes
    std::vector<TreeNodeIndex, Alloc> childOffsets_;
    //! @brief stores the parent index for every group of 8 sibling nodes, length the (numNodes - 1) / 8
    std::vector<TreeNodeIndex, Alloc> parents_;
    //! @brief store the first node index of every tree level, length = maxTreeLevel + 2
    std::vector<TreeNodeIndex, Alloc> levelRange_;

    //! @brief maps between the (level-key) sorted layout B and the unsorted intermediate binary layout A
    std::vector<TreeNodeIndex, Alloc> nodeOrder_;
    std::vector<TreeNodeIndex, Alloc> inverseNodeOrder_;

    //! @brief the cornerstone leaf SFC key array
    std::vector<KeyType> cstoneTree_;
};

template<class T, class KeyType, class CombinationFunction>
void upsweep(const Octree<KeyType>& octree, T* quantities, CombinationFunction combinationFunction)
{
    int currentLevel = maxTreeLevel<KeyType>{};

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

//! @brief perform upsweep, initializing leaf quantities from a separate array
template<class T, class KeyType, class CombinationFunction>
void upsweep(const Octree<KeyType>& octree,
             gsl::span<const T> leafQuantities,
             gsl::span<T> quantities,
             CombinationFunction combinationFunction)
{
    assert(leafQuantities.ssize() == octree.numLeafNodes());
    #pragma omp parallel for schedule(static)
    for (TreeNodeIndex i = 0; i < leafQuantities.ssize(); ++i)
    {
        TreeNodeIndex internalIdx = octree.toInternal(i);
        quantities[internalIdx]   = leafQuantities[i];
    }
    upsweep(octree, quantities.data(), combinationFunction);
}

template<class T, class KeyType>
void upsweepSum(const Octree<KeyType>& octree, gsl::span<const T> leafQuantities, gsl::span<T> quantities)
{
    auto sumFunction = [](auto a, auto b, auto c, auto d, auto e, auto f, auto g, auto h)
    { return a + b + c + d + e + f + g + h; };
    upsweep(octree, leafQuantities, quantities, sumFunction);
}

} // namespace cstone
