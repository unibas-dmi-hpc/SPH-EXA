#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "sfc/mortoncode.hpp"

namespace sphexa
{

/*! \brief returns the number of nodes in a tree
 *
 * \tparam I           32- or 64-bit unsigned integer type
 * \param tree         input tree
 * \return             the number of nodes
 *
 * This makes it explicit that a vector of n Morton codes
 * corresponds to a tree with n-1 nodes.
 */
template<class I>
std::size_t nNodes(const std::vector<I>& tree)
{
    assert(tree.size());
    return tree.size() - 1;
}

/*! \brief check whether octree invariants are fulfilled
 *
 * \tparam I           32- or 64-bit unsigned integer type
 * \param tree         octree nodes given as Morton codes of length @a nNodes+1
 * \param nNodes       number of nodes
 * \return             true if invariants ar satisfied, false otherwise
 *
 * The invariants are:
 *      - tree contains root node with code 0 and the maximum code (nodeRange<I>(0))
 *      - tree is sorted
 *      - difference between consecutive elements must be a power of 8
 *
 * These three conditions guarantee that
 *      - the entire space is covered
 *      - for each node, all its siblings (nodes at the same subdivision level)
 *        are present in the tree as a Morton code
 */
template<class I>
bool checkOctreeInvariants(const I* tree, int nNodes)
{
    // the root node delineated by code 0 and nodeRange<I>(0)
    // must be part of the tree
    if (nNodes < 1)
        return false;
    if (tree[0] != 0 || tree[nNodes] != nodeRange<I>(0))
        return false;

    for (int i = 0; i < nNodes; ++i)
    {
        if (i+1 < nNodes && tree[i] >= tree[i+1])
            return false;

        I range = tree[i+1] - tree[i];

        if (range == 0)
            return false;

        if (!isPowerOf8(range))
            return false;
    }

    return true;
}

/*! \brief count number of particles in each octree node
 *
 * \tparam I           32- or 64-bit unsigned integer type
 * \param tree         octree nodes given as Morton codes of length @a nNodes+1
 *                     needs to satisfy the octree invariants
 * \param counts       output particle counts per node, length = @a nNodes
 * \param nNodes       number of nodes in tree
 * \param codesStart   Morton code range start of particles to count
 * \param codesEnd     Morton code range end of particles to count
 */
template<class I>
void computeNodeCounts(const I* tree, std::size_t* counts, int nNodes, const I* codesStart, const I* codesEnd)
{
    for (int i = 0; i < nNodes; ++i)
    {
        I nodeStart = tree[i];
        I nodeEnd   = tree[i+1];

        // count particles in range
        auto rangeStart = std::lower_bound(codesStart, codesEnd, nodeStart);
        auto rangeEnd   = std::lower_bound(codesStart, codesEnd, nodeEnd);
        counts[i] = std::distance(rangeStart, rangeEnd);
    }
}

/*! \brief split or fuse octree nodes based on node counts relative to bucketSize
 *
 * \tparam I               32- or 64-bit unsigned integer type
 * \param[in] tree         octree nodes given as Morton codes of length @a nNodes
 *                         needs to satisfy the octree invariants
 * \param[in] counts       output particle counts per node, length = @a nNodes
 * \param[in] nNodes       number of nodes in tree
 * \param[in] bucketSize   maximum particle count per (leaf) node and
 *                         minimum particle count (strictly >) for (implicit) internal nodes
 * \param[out] converged   optional boolean flag to indicate convergence
 * \return                 the rebalanced Morton code octree
 */
template<class I>
std::vector<I> rebalanceTree(const I* tree, const std::size_t* counts, int nNodes,
                             int bucketSize, bool* converged = nullptr)
{
    std::vector<I> balancedTree;
    balancedTree.reserve(nNodes + 1);

    int changes = 0;

    int i = 0;
    while(i < nNodes)
    {
        I thisNode     = tree[i];
        I range        = tree[i+1] - thisNode;
        unsigned level = treeLevel(range);

        if (counts[i] > bucketSize && level < maxTreeLevel<I>{})
        {
            // split
            for (int sibling = 0; sibling < 8; ++sibling)
            {
                balancedTree.push_back(thisNode + sibling * nodeRange<I>(level + 1));
            }
            changes++;
            i++;
        }
        else if (level > 0 && // level 0 cannot be fused
                 parentIndex(thisNode, level) == 0 &&  // current node is first of 8 siblings
                 tree[i+8] == thisNode + nodeRange<I>(level - 1) && // next 7 nodes are all siblings
                 std::accumulate(counts + i, counts + i + 8, std::size_t(0)) <= bucketSize) // parent count too small
        {
            // fuse, by omitting the 7 siblings
            balancedTree.push_back(thisNode);
            changes++;
            i += 8;
        }
        else
        {
            // do nothing
            balancedTree.push_back(thisNode);
            i++;
        }
    }
    balancedTree.push_back(nodeRange<I>(0));

    if (converged != nullptr)
    {
        *converged = (changes == 0);
    }

    return balancedTree;
}

//! \brief returns an octree with just the root node
template<class I>
std::vector<I> makeRootNodeTree()
{
    std::vector<I> tree;

    tree.push_back(0);
    tree.push_back(nodeRange<I>(0));

    return tree;
}

//! \brief returns a uniform octree with 8^ceil(log8(nBuckets)) leaf nodes
template<class I>
std::vector<I> makeUniformNLevelTree(std::size_t nParticles, int bucketSize)
{
    // the minimum tree level needed is ceil(log8(nParticles/bucketSize))
    unsigned minTreeLevel = log8ceil(I(nParticles/std::size_t(bucketSize)));
    unsigned ticks        = 1u << minTreeLevel;

    std::vector<I> tree;
    tree.reserve(ticks*ticks*ticks + 1);

    // generate regular minTreeLevel tree
    for (unsigned x = 0; x < ticks; ++x)
        for (unsigned y = 0; y < ticks; ++y)
            for (unsigned z = 0; z < ticks; ++z)
            {
                tree.push_back(detail::codeFromBox<I>({x,y,z}, minTreeLevel));
            }

    tree.push_back(nodeRange<I>(0));

    sort(begin(tree), end(tree));

    return tree;
}


/*! \brief compute an octree from morton codes for a specified bucket size
 *
 * \tparam I           32- or 64-bit unsigned integer type
 * \param codesStart   particle morton code sequence start
 * \param codesEnd     particle morton code sequence end
 * \param bucketSize   maximum number of particles/codes per octree leaf node
 * \param[inout] tree  initial tree for the first iteration
 * \return             the tree and the node counts
 */
template<class I, class Reduce = void>
std::tuple<std::vector<I>, std::vector<std::size_t>>
computeOctree(const I* codesStart, const I* codesEnd, int bucketSize, std::vector<I>&& tree = std::vector<I>(0))
{
    if (!tree.size())
    {
        tree = makeUniformNLevelTree<I>(codesEnd - codesStart, bucketSize);
    }

    std::vector<std::size_t> counts(nNodes(tree));

    bool converged = false;
    while (!converged)
    {
        computeNodeCounts(tree.data(), counts.data(), nNodes(tree), codesStart, codesEnd);
        if constexpr (!std::is_same_v<void, Reduce>) Reduce{}(counts);
        std::vector<I> balancedTree;
        balancedTree = rebalanceTree(tree.data(), counts.data(), nNodes(tree), bucketSize, &converged);

        swap(tree, balancedTree);
        counts.resize(nNodes(tree));
    }

    return std::make_tuple(tree, counts);
}

/*! \brief Compute the maximum value of a given input array for each node in the global or local octree
 *
 * Example: For each node, compute the maximum smoothing length of all particles in that node
 *
 * \tparam I           32- or 64-bit unsigned integer type
 * \param tree         octree nodes given as Morton codes of length @a nNodes+1
 *                     This function does not rely on octree invariants, sortedness of the nodes
 *                     is the only requirement.
 * \param nNodes       number of nodes in tree
 * \param codesStart   sorted Morton code range start of particles to count
 * \param codesEnd     sorted Morton code range end of particles to count
 * \param ordering     Access input according to \a ordering
 *                     The sequence input[ordering[i]], i=0,...,N must list the elements of input
 *                     (i.e. the smoothing lengths) such that input[i] is a property of the particle
 *                     (x[i], y[i], z[i]), with x,y,z sorted according to Morton ordering.
 * \param input        Array to compute maximum over nodes, length = codesEnd - codesStart
 * \param output       maximum per node, length = @a nNodes
 */
template<class I, class T>
void computeNodeMax(const I* tree, int nNodes, const I* codesStart, const I* codesEnd,
                    const int* ordering, const T* input, T* output)
{
    for (int i = 0; i < nNodes; ++i)
    {
        I nodeStart = tree[i];
        I nodeEnd   = tree[i+1];

        // find elements belonging to particles in node i
        int startIndex = std::lower_bound(codesStart, codesEnd, nodeStart) - codesStart;
        int endIndex   = std::lower_bound(codesStart, codesEnd, nodeEnd)   - codesStart;

        T nodeMax = -INFINITY;
        for(int p = startIndex; p < endIndex; ++p)
        {
            T nodeElement = input[ordering[p]];
            if (nodeElement > nodeMax)
                nodeMax = nodeElement;
        }

        output[i] = nodeMax;
    }
}

//////////////////////////////////////////////////////////
// Tested, but not currently used functionality
// the code below implements an alternative octree format
// where each node has an upper and lower bound.
// Therefore, this format allows to have holes in the space filling curve
// of morton codes - or, in octree language, empty nodes can be omitted,
// such that not all the siblings of a node in the tree are guaranteed to exist.
//////////////////////////////////////////////////////////

//! \brief Defines data to describe an octree node
template<class I>
struct SfcNode
{
    /*! The morton start and end codes define the scope of the node.
     *  They are equivalent to an integer (ix,iy,iz) index triple that describes the
     *  spatial location of the box plus the octree division level (or the size of the box)
     */
    I startCode;
    I endCode;

    /*! The particle content of the node.
     *
     *  coordinateIndex: Stores the index of the first morton code in the full array
     *                   of sorted morton codes that falls into the node.
     *  count:           Number of morton codes in this node. Since the morton code
     *                   array is assumed to be sorted, all particles in the range
     *                   [coordinateIndex, coordinateIndex + count] fall into the node.
     *
     *  Both indices are also valid for the x,y,z coordinate arrays, provided that they
     *  are sorted according to the ascending morton code ordering.
     */
    unsigned coordinateIndex;
    unsigned count;
};

template<class I>
inline bool operator<(const SfcNode<I>& lhs, const SfcNode<I>& rhs)
{
    return std::tie(lhs.startCode, lhs.endCode) < std::tie(rhs.startCode, rhs.endCode);
}

template<class I>
inline bool operator==(const SfcNode<I>& lhs, const SfcNode<I>& rhs)
{
    return std::tie(lhs.startCode, lhs.endCode, lhs.coordinateIndex, lhs.count)
           == std::tie(rhs.startCode, rhs.endCode, rhs.coordinateIndex, rhs.count);

}

//! \brief Defines data to describe an octree node, no coordinate reference
template<class I>
struct GlobalSfcNode
{
    GlobalSfcNode(I start, I end, [[maybe_unused]] unsigned ignore, std::size_t c)
        : startCode(start), endCode(end), count(c) { }

    //! start and end codes
    I startCode;
    I endCode;

    //! The particle content of the node.
    std::size_t count;
};

template<class I>
inline bool operator<(const GlobalSfcNode<I>& lhs, const GlobalSfcNode<I>& rhs)
{
    return std::tie(lhs.startCode, lhs.endCode) < std::tie(rhs.startCode, rhs.endCode);
}

template<class I>
inline bool operator==(const GlobalSfcNode<I>& lhs, const GlobalSfcNode<I>& rhs)
{
    return std::tie(lhs.startCode, lhs.endCode, lhs.count)
           == std::tie(rhs.startCode, rhs.endCode, rhs.count);

}

/*! \brief aggregate mortonCodes into octree leaf nodes of increased size
 *
 * \tparam NodeType       SfcNode, either with or without offset into coordinate arrays
 * \tparam I              32- or 64-bit unsigned integer
 * \param[in] mortonCodes input mortonCode array
 * \param[in] bucketSize  determine size of octree nodes such that
 *                        (leaf node).count <= bucketSize
 *                        and for their parents (<=> internal nodes)
 *                        (parent node).count > bucketSize
 *
 * \return vector with the sorted octree leaf nodes
 */
template<template<class> class NodeType, class I>
std::vector<NodeType<I>> trimZCurve(const std::vector<I>& mortonCodes, unsigned bucketSize)
{
    std::vector<SfcNode<I>> ret;

    unsigned n = mortonCodes.size();
    unsigned i = 0;

    I previousBoxEnd = 0;

    while (i < n)
    {
        I code = mortonCodes[i];

        // the smallest code more than bucketSize away
        // need to find a box that stays below it
        I codeLimit = (i + bucketSize < n) ? mortonCodes[i + bucketSize] : nodeRange<I>(0);

        // find smallest j in [i, i + bucketSize], such that codeLimit < get<1>(smallestCommonBox(mCodes[i], mCodes[j]))
        auto isInBox = [code](I c1_, I c2_){ return c1_ < std::get<1>(smallestCommonBox(code, c2_)); };
        auto jIt = std::upper_bound(cbegin(mortonCodes) + i, cbegin(mortonCodes) + std::min(n, i + bucketSize), codeLimit, isInBox);
        unsigned j = jIt - cbegin(mortonCodes);

        // find smallest k in [i, i + bucketSize], such that not(get<0>(smallestCommonBox(mCodes[i], mCodes[k])) < previousBoxEnd)
        auto isBelowBox = [code](I c1_, I c2_){ return !(std::get<0>(smallestCommonBox(code, c1_)) < c2_); };
        auto kIt = std::lower_bound(cbegin(mortonCodes) + i, cbegin(mortonCodes) + std::min(n, i + bucketSize), previousBoxEnd, isBelowBox);
        unsigned k = kIt - cbegin(mortonCodes);

        // the smaller of the two indices is the one that produces a range of codes
        // with an enclosing octree box that both doesn't overlap with the previous one
        // and does not include more than bucketSize particles
        j = std::min(j, k);

        std::tuple<I, I> box = smallestCommonBox(code, mortonCodes[j-1]);
        ret.push_back(NodeType<I>{std::get<0>(box), std::get<1>(box), i, j-i});
        i = j;
        previousBoxEnd = std::get<1>(box);
    }

    return ret;
}

} // namespace sphexa
