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


/*! @brief atomically update a maximum value and return the previous maximum value
 *
 * @tparam       T               integer type
 * @param[inout] maximumValue    the maximum value to be atomically updated
 * @param[in]    newValue        the value with which to compute the new maximum
 * @return                       the previous maximum value
 *
 * Lifted into global namespace to enable correct overload resolution with CUDA builtin atomicMax
 * in device code.
 */
template<typename T>
inline T atomicMax(std::atomic<T>* maximumValue, const T& newValue) noexcept
{
    T previousValue = *maximumValue;
    while(previousValue < newValue && !maximumValue->compare_exchange_weak(previousValue, newValue))
    {}
    return previousValue;
}

namespace cstone
{

/*! @brief octree node for the internal part of cornerstone octrees
 *
 * @tparam KeyType  32- or 64 bit unsigned integer
 */
template<class KeyType>
struct OctreeNode
{
    /*! @brief the Morton code prefix
     *
     * Shared among all the node's children. Only the first prefixLength bits are relevant.
     */
    KeyType   prefix;

    //! @brief octree division level, equals 1/3rd the number of bits in prefix to interpret
    int level;

    //! @brief internal node index of the parent node
    TreeNodeIndex parent;

    /*! @brief Child node indices
     *
     *  If isLeafIndex(child[i]) is true, loadLeafIndex(child[i]) is a leaf node index.
     *  Otherwise, child[i] is the index of an octree leaf node.
     *  Note that the indices in these two cases refer to two different arrays!
     */
    TreeNodeIndex child[8];

    friend bool operator==(const OctreeNode<KeyType>& lhs, const OctreeNode<KeyType>& rhs)
    {
        bool eqChild = true;
        for (int i = 0; i < 8; ++i)
        {
            eqChild = eqChild && lhs.child[i] == rhs.child[i];
        }

        return lhs.prefix == rhs.prefix &&
               lhs.level  == rhs.level &&
               lhs.parent == rhs.parent &&
               eqChild;
    }
};

template<class KeyType, class AtomicInteger>
HOST_DEVICE_FUN void nodeDepthElement(TreeNodeIndex i, const OctreeNode<KeyType>* octree, AtomicInteger* depths)
{
    int nLeafChildren = 0;
    for (int octant = 0; octant < 8; ++octant)
    {
        if (isLeafIndex(octree[i].child[octant]))
        {
            nLeafChildren++;
        }
    }

    if (nLeafChildren == 8) { depths[i] = 1; } // all children are leaves - maximum depth is 1
    else                    { return; } // another thread will climb the tree and set the depth

    TreeNodeIndex nodeIndex = i;
    int depth = 1;

    // race to the top
    do
    {   // ascend one level
        nodeIndex = octree[nodeIndex].parent;
        depth++;

        // set depths[nodeIndex] = max(depths[nodeIndex], depths) and store previous value
        // of depths[nodeIndex] in previousMax
        int previousMax = atomicMax(depths + nodeIndex, depth);
        if (previousMax >= depth)
        {
            // another thread already set a higher value for depths[nodeIndex], drop out of race
            break;
        }

    } while (nodeIndex != octree[nodeIndex].parent);
}

/*! @brief calculate distance to farthest leaf for each internal node in parallel
 *
 * @tparam KeyType             32- or 64-bit integer type
 * @param[in]  octree    an octree, length @a nNodes
 * @param[in]  nNodes    number of (internal) nodes
 * @param[out] depths    array of length @a nNodes, contains
 *                       the distance to the farthest leaf for each node.
 *                       The distance is equal to 1 for each node whose children are only leaves.
 */
template<class KeyType>
void nodeDepth(const OctreeNode<KeyType>* octree, TreeNodeIndex nNodes, std::atomic<int>* depths)
{
    #pragma omp parallel for
    for (TreeNodeIndex i = 0; i < nNodes; ++i)
    {
        nodeDepthElement(i, octree, depths);
    }
}

/*! @brief calculates the tree node ordering for descending max leaf distance
 *
 * @tparam KeyType                    32- or 64-bit integer
 * @param[in]  octree           input array of OctreeNode<KeyType>
 * @param[in]  nNodes           number of input octree nodes
 * @param[out] ordering         output ordering, a permutation of [0:nNodes]
 * @param[out] nNodesPerLevel   number of nodes per value of farthest leaf distance
 *                              length is maxTreeLevel<KeyType>{} (10 or 21)
 *
 *  nNodesPerLevel[0] --> number of leaf nodes (NOT set in this function)
 *  nNodesPerLevel[1] --> number of nodes with maxDepth = 1, children: only leaves
 *  nNodesPerLevel[2] --> number of internal nodes with maxDepth = 2, children: leaves and maxDepth = 1
 *
 * First calculates the distance of the farthest leaf for each node, then
 * determines an ordering that sorts the nodes according to decreasing distance.
 * The resulting ordering is valid from a scatter-perspective, i.e.
 * Nodes at <oldIndex> should be moved to @a ordering[<oldIndex>].
 */
template<class KeyType>
void decreasingMaxDepthOrder(const OctreeNode<KeyType>* octree,
                             TreeNodeIndex nNodes,
                             TreeNodeIndex* ordering,
                             TreeNodeIndex* nNodesPerLevel)
{
    std::vector<std::atomic<int>> depths(nNodes);

    #pragma omp parallel for schedule(static)
    for (TreeNodeIndex i = 0; i < nNodes; ++i)
    {
        depths[i] = 0;
    }

    nodeDepth(octree, nNodes, depths.data());

    // inverse ordering will be the inverse permutation of the
    // output ordering and also corresponds to the ordering
    // from a gather-perspective
    std::vector<TreeNodeIndex> inverseOrdering(nNodes);
    #pragma omp parallel for schedule(static)
    for (TreeNodeIndex i = 0; i < nNodes; ++i)
    {
        inverseOrdering[i] = i;
    }

    std::vector<int> depths_v(begin(depths), end(depths));
    sort_by_key(begin(depths_v), end(depths_v), begin(inverseOrdering), std::greater<TreeNodeIndex>{});

    #pragma omp parallel for schedule(static)
    for (TreeNodeIndex i = 0; i < nNodes; ++i)
    {
        ordering[i] = i;
    }

    sort_by_key(begin(inverseOrdering), end(inverseOrdering), ordering);

    // count nodes per value of depth
    for (unsigned depth = 1; depth < maxTreeLevel<KeyType>{}; ++depth)
    {
        auto it1 = std::lower_bound(begin(depths_v), end(depths_v), depth, std::greater<TreeNodeIndex>{});
        auto it2 = std::upper_bound(begin(depths_v), end(depths_v), depth, std::greater<TreeNodeIndex>{});
        nNodesPerLevel[depth] = TreeNodeIndex(std::distance(it1, it2));
    }
}

/*! @brief reorder internal octree nodes according to a map
 *
 * @tparam KeyType                   32- or 64-bit unsigned integer
 * @param[in]  oldNodes        array of octree nodes, length @a numInternalNodes
 * @param[in]  rewireMap       a permutation of [0:numInternalNodes]
 * @param[in]  nInternalNodes  number of internal octree nodes
 * @param[out] newNodes        reordered array of octree nodes, length @a numInternalNodes
 */
template<class KeyType>
void rewireInternal(const OctreeNode<KeyType>* oldNodes,
                    const TreeNodeIndex* rewireMap,
                    TreeNodeIndex nInternalNodes,
                    OctreeNode<KeyType>* newNodes)
{
    #pragma omp parallel for schedule(static)
    for (TreeNodeIndex oldIndex = 0; oldIndex < nInternalNodes; ++oldIndex)
    {
        // node at <oldIndex> moves to <newIndex>
        TreeNodeIndex newIndex = rewireMap[oldIndex];

        OctreeNode<KeyType> newNode = oldNodes[oldIndex];
        newNode.parent = rewireMap[newNode.parent];
        for (int octant = 0; octant < 8; ++octant)
        {
            if (!isLeafIndex(newNode.child[octant]))
            {
                TreeNodeIndex oldChild = newNode.child[octant];
                newNode.child[octant]  = rewireMap[oldChild];
            }
        }

        newNodes[newIndex] = newNode;
    }
}

/*! @brief translate each input index according to a map
 *
 * @tparam Index         integer type
 * @param[in]  input     input indices, length @a nElements
 * @param[in]  rewireMap index translation map
 * @param[in]  nElements number of Elements
 * @param[out] output    translated indices, length @a nElements
 */
template<class Index>
void rewireIndices(const Index* input,
                   const Index* rewireMap,
                   size_t nElements,
                   Index* output)
{
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < nElements; ++i)
    {
        output[i] = rewireMap[input[i]];
    }
}

/*! @brief construct the internal octree node with index @p nodeIndex
 *
 * @tparam KeyType                   32- or 64-bit unsigned integer type
 * @param[out]  internalOctree       linear array of OctreeNode<KeyType>'s
 * @param[in]   binaryTree           linear array of binary tree nodes
 * @param[in]   nodeIndex            element of @p internalOctree to construct
 * @param[in]   octreeToBinaryIndex  octreeToBinaryIndex[i] stores the index of the binary node in
 *                                   @p binaryTree with the identical prefix as the octree node with index i
 * @param[in]   binaryToOctreeIndex  inverts @p octreeToBinaryIndex
 * @param[out]  leafParents[out]     linear array of indices to store the parent index of each octree leaf
 *                                   number of elements corresponds to the number of nodes in the cornerstone
 *                                   octree that was used to construct @p binaryTree
 *
 * This function sets all members of internalOctree[nodeIndex] except the parent member.
 * (Exception: the parent of the root node is set to 0)
 * In addition, it sets the parent member of the child nodes to @p nodeIndex.
 */
template<class KeyType>
HOST_DEVICE_FUN inline
void constructOctreeNode(OctreeNode<KeyType>*       internalOctree,
                         const BinaryNode<KeyType>* binaryTree,
                         TreeNodeIndex        nodeIndex,
                         const TreeNodeIndex* scatterMap,
                         const TreeNodeIndex* binaryToOctreeIndex,
                         TreeNodeIndex*       leafParents)
{
    OctreeNode<KeyType>& octreeNode = internalOctree[nodeIndex];

    TreeNodeIndex bi  = scatterMap[nodeIndex]; // binary tree index
    octreeNode.prefix = decodePlaceholderBit(binaryTree[bi].prefix);
    octreeNode.level  = decodePrefixLength(binaryTree[bi].prefix) / 3;

    // the root node is its own parent
    if (octreeNode.level == 0)
    {
        octreeNode.parent = 0;
    }

    for (int hx = 0; hx < 2; ++hx)
    {
        for (int hy = 0; hy < 2; ++hy)
        {
            for (int hz = 0; hz < 2; ++hz)
            {
                int octant = 4*hx + 2*hy + hz;
                TreeNodeIndex childBinaryIndex =
                    binaryTree[binaryTree[binaryTree[bi].child[hx]].child[hy]].child[hz];

                if (!isLeafIndex(childBinaryIndex))
                {
                    TreeNodeIndex childOctreeIndex = binaryToOctreeIndex[childBinaryIndex];
                    octreeNode.child[octant]       = childOctreeIndex;

                    internalOctree[childOctreeIndex].parent = nodeIndex;
                }
                else
                {
                    TreeNodeIndex octreeLeafIndex = loadLeafIndex(childBinaryIndex);
                    octreeNode.child[octant]      = storeLeafIndex(octreeLeafIndex);
                    leafParents[octreeLeafIndex]  = nodeIndex;
                }
            }
        }
    }
}

/*! @brief translate an internal binary radix tree into an internal octree
 *
 * @tparam KeyType                   32- or 64-bit unsigned integer
 * @param[in]  binaryTree      binary tree nodes
 * @param[in]  nLeafNodes      number of octree leaf nodes used to construct @p binaryTree
 * @param[out] internalOctree  output internal octree nodes, length = (@p numLeafNodes-1) / 7
 * @param[out] leafParents     node index of the parent node for each leaf, length = @p numLeafNodes
 *
 */
template<class KeyType>
void createInternalOctreeCpu(const BinaryNode<KeyType>* binaryTree, TreeNodeIndex nLeafNodes,
                             OctreeNode<KeyType>* internalOctree, TreeNodeIndex* leafParents)
{
    // we ignore the last binary tree node which is a duplicate root node
    TreeNodeIndex nBinaryNodes = nLeafNodes - 1;

    // one extra element to store the total sum of the exclusive scan
    std::vector<TreeNodeIndex> prefixes(nBinaryNodes + 1);
    #pragma omp parallel for schedule(static)
    for (TreeNodeIndex i = 0; i < nBinaryNodes; ++i)
    {
        int  prefixLength = decodePrefixLength(binaryTree[i].prefix);
        bool divisibleBy3 = prefixLength % 3 == 0;
        prefixes[i] = (divisibleBy3) ? 1 : 0;
    }

    // stream compaction: scan and scatter
    exclusiveScan(prefixes.data(), prefixes.size());

    // nInternalOctreeNodes is also equal to prefixes[nBinaryNodes]
    TreeNodeIndex nInternalOctreeNodes = (nLeafNodes-1)/7;
    std::vector<TreeNodeIndex> scatterMap(nInternalOctreeNodes);

    // compaction step, scatterMap -> compacted list of binary nodes that correspond to octree nodes
    #pragma omp parallel for schedule(static)
    for (TreeNodeIndex i = 0; i < nBinaryNodes; ++i)
    {
        bool isOctreeNode = (prefixes[i+1] - prefixes[i]) == 1;
        if (isOctreeNode)
        {
            int octreeNodeIndex = prefixes[i];
            scatterMap[octreeNodeIndex] = i;
        }
    }

    #pragma omp parallel for schedule(static)
    for (TreeNodeIndex i = 0; i < nInternalOctreeNodes; ++i)
    {
        constructOctreeNode(internalOctree, binaryTree, i, scatterMap.data(), prefixes.data(), leafParents);
    }
}


/*! @brief This class unifies a cornerstone octree with the internal part
 *
 * @tparam KeyType          32- or 64-bit unsigned integer
 *
 * Leaves are stored separately from the internal nodes. For either type of node, just a single buffer is allocated.
 * All nodes are guaranteed to be stored ordered according to decreasing value of the distance of the farthest leaf.
 * This property is relied upon by the generic upsweep implementation.
 */
template<class KeyType>
class Octree {
public:
    Octree() : nNodesPerLevel_(maxTreeLevel<KeyType>{}) {}

    /*! @brief sets the leaves to the provided ones and updates the internal part based on them
     *
     * @param firstLeaf  first leaf
     * @param lastLeaf   last leaf
     *
     * internal state change:
     *      -full tree update
     */
    template<class InputIterator>
    void update(InputIterator firstLeaf, InputIterator lastLeaf)
    {
        assert(lastLeaf > firstLeaf);

        // make space for leaf nodes
        TreeNodeIndex treeSize = lastLeaf - firstLeaf;
        cstoneTree_.resize(treeSize);
        std::copy(firstLeaf, lastLeaf, cstoneTree_.data());

        updateInternalTree();
    }

    void update(std::vector<KeyType>&& newLeaves)
    {
        cstoneTree_ = std::move(newLeaves);
        updateInternalTree();
    }

    //! @brief total number of nodes in the tree
    [[nodiscard]] inline TreeNodeIndex numTreeNodes() const
    {
        return nNodes(cstoneTree_) + internalTree_.size();
    }

    /*! @brief number of nodes with given value of maxDepth
     *
     * @param[in] maxDepth  distance to farthest leaf
     * @return              number of nodes in the tree with given value for maxDepth
     *
     * Some relations with other node-count functions:
     *      numTreeNodes(0) == numLeafNodes()
     *      sum([numTreeNodes(i) for i in [0:maxTreeLevel<KeyType>{}]]) == numTreeNodes()
     *      sum([numTreeNodes(i) for i in [1:maxTreeLevel<KeyType>{}]]) == numInternalNodes()
     */
    [[nodiscard]] inline TreeNodeIndex numTreeNodes(unsigned maxDepth) const
    {
        assert(maxDepth < maxTreeLevel<KeyType>{});
        return nNodesPerLevel_[maxDepth];
    }

    //! @brief number of leaf nodes in the tree
    [[nodiscard]] inline TreeNodeIndex numLeafNodes() const
    {
        return nNodes(cstoneTree_);
    }

    //! @brief number of internal nodes in the tree, equal to (numLeafNodes()-1) / 7
    [[nodiscard]] inline TreeNodeIndex numInternalNodes() const
    {
        return internalTree_.size();
    }

    /*! @brief check whether node is a leaf
     *
     * @param[in] node    node index, range [0:numTreeNodes()]
     * @return            true or false
     */
    [[nodiscard]] inline bool isLeaf(TreeNodeIndex node) const
    {
        assert(node < numTreeNodes());
        return node >= TreeNodeIndex(internalTree_.size());
    }

    /*! @brief convert a leaf index (indexed from first leaf starting from 0) to 0-indexed from root
     *
     * @param[in] node    leaf node index, range [0:numLeafNodes()]
     * @return            octree index, relative to the root node
     */
    [[nodiscard]] inline TreeNodeIndex toInternal(TreeNodeIndex node) const
    {
        return node + numInternalNodes();
    }

    /*! @brief check whether child of node is a leaf
     *
     * @param[in] node    node index, range [0:numInternalNodes()]
     * @param[in] octant  octant index, range [0:8]
     * @return            true or false
     *
     * If @p node is not internal, behavior is undefined.
     */
    [[nodiscard]] inline bool isLeafChild(TreeNodeIndex node, int octant) const
    {
        assert(node < TreeNodeIndex(internalTree_.size()));
        return isLeafIndex(internalTree_[node].child[octant]);
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
        assert(node < TreeNodeIndex(internalTree_.size()));

        TreeNodeIndex childIndex = internalTree_[node].child[octant];
        if (isLeafIndex(childIndex))
        {
            childIndex = loadLeafIndex(childIndex) + numInternalNodes();
        }

        return childIndex;
    }

    /*! @brief return child node indices with leaf indices starting from 0
     *
     * @param[in] node    node index, range [0:numInternalNodes()]
     * @param[in] octant  octant index, range [0:8]
     * @return            child node index, range [0:numInternalNodes()] if child is internal
     *                    or [0:numLeafNodes()] if child is a leaf
     *
     * If @a node is not internal, behavior is undefined.
     * Note: the indices returned by this function refer to two different arrays, depending on
     * whether the child specified by @p node and @p octant is a leaf or an internal node.
     */
    [[nodiscard]] inline TreeNodeIndex childDirect(TreeNodeIndex node, int octant) const
    {
        TreeNodeIndex childIndex = internalTree_[node].child[octant];
        if (isLeafIndex(childIndex))
        {
            return loadLeafIndex(childIndex);
        }
        else
        {
            return childIndex;
        }
    }

    /*! @brief index of parent node
     *
     * Note: the root node is its own parent
     */
    [[nodiscard]] inline TreeNodeIndex parent(TreeNodeIndex node) const
    {
        if (node < numInternalNodes())
        {
            return internalTree_[node].parent;
        } else
        {
            return leafParents_[node - numInternalNodes()];
        }
    }

    //! @brief lowest SFC key contained int the geometrical box of @p node
    [[nodiscard]] inline KeyType codeStart(TreeNodeIndex node) const
    {
        if (node < numInternalNodes())
        {
            return internalTree_[node].prefix;
        } else
        {
            return cstoneTree_[node - numInternalNodes()];
        }
    }

    //! @brief highest SFC key contained in the geometrical box of @p node
    [[nodiscard]] inline KeyType codeEnd(TreeNodeIndex node) const
    {
        if (node < numInternalNodes())
        {
            return internalTree_[node].prefix + nodeRange<KeyType>(internalTree_[node].level);
        } else
        {
            return cstoneTree_[node - numInternalNodes() + 1];
        }
    }

    /*! @brief octree subdivision level for @p node
     *
     * Returns 0 for the root node. Highest value is maxTreeLevel<KeyType>{}.
     */
    [[nodiscard]] inline int level(TreeNodeIndex node) const
    {
        if (node < numInternalNodes())
        {
            return internalTree_[node].level;
        } else
        {
            return treeLevel(cstoneTree_[node - numInternalNodes() + 1] -
                             cstoneTree_[node - numInternalNodes()]);
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
        TreeNodeIndex nodeIdx = 0;
        if (codeStart(nodeIdx) == startKey && codeStart(nodeIdx) == endKey) { return nodeIdx; }

        if (isLeaf(nodeIdx)) { return numTreeNodes(); } // not found

        // nodeIdx is internal
        while (internalTree_[nodeIdx].prefix != startKey)
        {
            nodeIdx = refineIdx(nodeIdx, startKey);
            if (isLeafIndex(nodeIdx))
            {
                if (cstoneTree_[loadLeafIndex(nodeIdx)] != startKey ||
                    cstoneTree_[loadLeafIndex(nodeIdx) + 1] != endKey)
                {
                    // not found
                    return numTreeNodes();
                }

                else { return loadLeafIndex(nodeIdx) + numInternalNodes(); }
            }
        }

        // nodeIdx is still internal
        while (internalTree_[nodeIdx].prefix + nodeRange<KeyType>(internalTree_[nodeIdx].level) != endKey)
        {
            nodeIdx = internalTree_[nodeIdx].child[0];
            if (isLeafIndex(nodeIdx))
            {
                if (cstoneTree_[loadLeafIndex(nodeIdx) + 1] != endKey) { return numTreeNodes(); }
                else { return loadLeafIndex(nodeIdx) + numInternalNodes(); }
            }
        }

        return nodeIdx;
    }

    [[nodiscard]] gsl::span<const KeyType> treeLeaves() const
    {
        return cstoneTree_;
    }

    [[nodiscard]] const TreeNodeIndex* leafParents() const
    {
        return leafParents_.data();
    }

private:

    /*! @brief find the child that contains the given key
     *
     * @param nodeIdx has to be internal
     * @param key     SFC key to look for
     * @return        the tree node index of the child of nodeIdx that contains key
     */
    TreeNodeIndex refineIdx(TreeNodeIndex nodeIdx, KeyType key) const
    {
        for (int octant = 1; octant < 8; ++octant)
        {
            TreeNodeIndex child = internalTree_[nodeIdx].child[octant];
            KeyType nodeStart = (isLeafIndex(child)) ? cstoneTree_[loadLeafIndex(child)] :
                                                       internalTree_[child].prefix;
            if (key < nodeStart) { return internalTree_[nodeIdx].child[octant - 1]; }
        }
        return internalTree_[nodeIdx].child[7];
    }

    //! @brief regenerates the internal tree based on (a changed) cstoneTree_
    void updateInternalTree()
    {
        nNodesPerLevel_[0] = nNodes(cstoneTree_);

        binaryTree_.resize(nNodes(cstoneTree_));
        createBinaryTree(cstoneTree_.data(), nNodes(cstoneTree_), binaryTree_.data());

        std::vector<OctreeNode<KeyType>> preTree((nNodes(cstoneTree_) - 1) / 7);
        std::vector<TreeNodeIndex> preLeafParents(nNodes(cstoneTree_));

        createInternalOctreeCpu(binaryTree_.data(), nNodes(cstoneTree_), preTree.data(), preLeafParents.data());

        // re-sort internal nodes to establish a max-depth ordering
        std::vector<TreeNodeIndex> ordering(preTree.size());
        // determine ordering
        decreasingMaxDepthOrder(preTree.data(), preTree.size(), ordering.data(), nNodesPerLevel_.data());
        // apply the ordering to the internal tree;
        internalTree_.resize(preTree.size());
        rewireInternal(preTree.data(), ordering.data(), preTree.size(), internalTree_.data());

        // apply ordering to leaf parents
        leafParents_.resize(preLeafParents.size());

        // internal tree is empty if a single leaf node is also the tree-root
        if (!internalTree_.empty())
        {
            leafParents_[0] = 0;
            rewireIndices(preLeafParents.data(), ordering.data(), preLeafParents.size(), leafParents_.data());
        }
    }

    //! @brief cornerstone octree, just the leaves
    std::vector<KeyType>       cstoneTree_;

    //! @brief indices into internalTree_ to store the parent index of each leaf
    std::vector<TreeNodeIndex> leafParents_;

    //! @brief the internal tree
    std::vector<OctreeNode<KeyType>> internalTree_;

    /*! @brief the internal part as binary radix nodes, precursor to internalTree_
     *
     * This is kept here because the binary format is faster for findHalos / collision detection
     */
    std::vector<BinaryNode<KeyType>> binaryTree_;

    /*! @brief stores the number of nodes for each of the maxTreeLevel<KeyType>{} possible values of
     *
     *  maxDepth is the distance of the farthest leaf, i.e.
     *  nNodesPerLevel[0] --> number of leaf nodes
     *  nNodesPerLevel[1] --> number of nodes with maxDepth = 1, children: only leaves
     *  nNodesPerLevel[2] --> number of internal nodes with maxDepth = 2, children: leaves and maxDepth = 1
     *  etc.
     */
     std::vector<TreeNodeIndex> nNodesPerLevel_;
};

} // namespace cstone
