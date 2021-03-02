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

/*! \file
 * \brief Generation of local and global octrees in cornerstone format
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * In the cornerstone format, the octree is stored as sequence of Morton codes
 * fulfilling three invariants. Each code in the sequence both signifies the
 * the start Morton code of an octree leaf node and serves as an upper Morton code bound
 * for the previous node.
 *
 * The invariants of the cornerstone format are:
 *      - code sequence contains code 0 and the maximum code 2^30 or 2^61
 *      - code sequence is sorted by ascending code value
 *      - difference between consecutive elements must be a power of 8
 *
 * The consequences of these invariants are:
 *      - the entire space is covered, i.e. there are no holes in the tree
 *      - only leaf nodes are stored
 *      - for each leaf node, all its siblings (nodes at the same subdivision level with
 *        the same parent) are present in the Morton code sequence
 *      - each node with index i is defined by its lowest possible Morton code at position
 *        i in the vector and the highest possible (excluding) Morton code at position i+1
 *        in the vector
 *      - a vector of length N represents N-1 leaf nodes
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <numeric>
#include <vector>
#include <tuple>

#include "cstone/sfc/mortoncode.hpp"
#include "cstone/primitives/scan.hpp"

namespace cstone
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

/*! \brief count number of particles in each octree node
 *
 * \tparam I           32- or 64-bit unsigned integer type
 * \param tree         octree nodes given as Morton codes of length @a nNodes+1
 *                     needs to satisfy the octree invariants
 * \param counts       output particle counts per node, length = @a nNodes
 * \param nNodes       number of nodes in tree
 * \param codesStart   Morton code range start of particles to count
 * \param codesEnd     Morton code range end of particles to count
 * \param maxCount     maximum particle count per node to store, this is used
 *                     to prevent overflow in MPI_Allreduce
 */
template<class I>
void computeNodeCounts(const I* tree, unsigned* counts, int nNodes, const I* codesStart, const I* codesEnd,
                       unsigned maxCount = std::numeric_limits<unsigned>::max())
{
    int firstNode = 0;
    int lastNode  = nNodes;
    if (codesStart != codesEnd)
    {
        firstNode = std::upper_bound(tree, tree + nNodes, *codesStart) - tree - 1;
        lastNode  = std::upper_bound(tree, tree + nNodes, *(codesEnd-1)) - tree;
    }

    #pragma omp parallel
    {
        #pragma omp for schedule(static) nowait
        for (int i = 0; i < firstNode; ++i)
            counts[i] = 0;

        #pragma omp for schedule(static) nowait
        for (int i = lastNode; i < nNodes; ++i)
            counts[i] = 0;

        #pragma omp for schedule(static)
        for (int i = firstNode; i < lastNode; ++i)
        {
            I nodeStart = tree[i];
            I nodeEnd   = tree[i+1];

            // count particles in range
            auto rangeStart = std::lower_bound(codesStart, codesEnd, nodeStart);
            auto rangeEnd   = std::lower_bound(codesStart, codesEnd, nodeEnd);
            unsigned count  = std::distance(rangeStart, rangeEnd);
            counts[i]       = std::min(count, maxCount);
        }
    }
}

/*! \brief Compute split or fuse decision for each octree node in parallel
 *
 * \tparam I               32- or 64-bit unsigned integer type
 * \param[in] tree         octree nodes given as Morton codes of length @a nNodes
 *                         needs to satisfy the octree invariants
 * \param[in] counts       output particle counts per node, length = @a nNodes
 * \param[in] nNodes       number of nodes in tree
 * \param[in] bucketSize   maximum particle count per (leaf) node and
 *                         minimum particle count (strictly >) for (implicit) internal nodes
 * \param[out] nodeOps     stores rebalance decision result for each node, length = @a nNodes
 * \return                 number of nodes split + fused
 *
 * For each node i in the tree, in nodeOps[i], stores
 *  - 0 if to be merged
 *  - 1 if unchanged,
 *  - 8 if to be split.
 */
template<class I, class LocalIndex>
int rebalanceDecision(const I* tree, const unsigned* counts, int nNodes,
                      unsigned bucketSize, LocalIndex* nodeOps)
{
    std::atomic<int> changes{0};

    #pragma omp parallel for
    for (int i = 0; i < nNodes; ++i)
    {
        I thisNode     = tree[i];
        I range        = tree[i+1] - thisNode;
        unsigned level = treeLevel(range);

        nodeOps[i] = 1; // default: do nothing

        if (counts[i] > bucketSize && level < maxTreeLevel<I>{})
        {
            changes++;
            nodeOps[i] = 8; // split
        }
        else if (level > 0) // level 0 cannot be fused
        {
            int pi = parentIndex(thisNode, level);
            assert (i >= pi);
            // node's 7 siblings are next to each other
            bool siblings = (tree[i-pi+8] == tree[i-pi] + nodeRange<I>(level - 1));
            if (siblings && pi > 0) // if not first of 8 siblings
            {
                size_t parentCount = std::accumulate(counts + i - pi, counts + i - pi + 8, size_t(0));
                if (parentCount <= bucketSize)
                {
                    nodeOps[i] = 0; // fuse
                    changes++;
                }
            }
        }
    }

    return changes;
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
std::vector<I> rebalanceTree(const I* tree, const unsigned* counts, int nNodes,
                             unsigned bucketSize, bool* converged = nullptr)
{
    using LocalIndex = unsigned;
    std::vector<LocalIndex> nodeOps(nNodes + 1);

    int changes = rebalanceDecision(tree, counts, nNodes, bucketSize, nodeOps.data());

    exclusiveScan(nodeOps.data(), nNodes + 1);

    std::vector<I> balancedTree(*nodeOps.rbegin() + 1);

    #pragma omp parallel for
    for (int i = 0; i < nNodes; ++i)
    {
        I thisNode     = tree[i];
        I range        = tree[i+1] - thisNode;
        unsigned level = treeLevel(range);

        LocalIndex opCode       = nodeOps[i+1] - nodeOps[i];
        LocalIndex newNodeIndex = nodeOps[i];

        if (opCode == 1)
        {
            balancedTree[newNodeIndex] = thisNode;
        }
        else if (opCode == 8)
        {
            for (int sibling = 0; sibling < 8; ++sibling)
            {
                balancedTree[newNodeIndex + sibling] = thisNode + sibling * nodeRange<I>(level + 1);
            }
        }
    }
    *balancedTree.rbegin() = nodeRange<I>(0);

    if (converged != nullptr)
    {
        *converged = (changes == 0);
    }

    return balancedTree;
}


/*! \brief compute an octree from morton codes for a specified bucket size
 *
 * \tparam I           32- or 64-bit unsigned integer type
 * \param codesStart   particle morton code sequence start
 * \param codesEnd     particle morton code sequence end
 * \param bucketSize   maximum number of particles/codes per octree leaf node
 * \param maxCount     if actual node counts are higher, they will be capped to \a maxCount
 * \param[inout] tree  initial tree for the first iteration
 * \return             the tree and the node counts
 *
 * Remarks:
 *    It is sensible to assume that the bucket size of the tree is much smaller than 2^32,
 *    and thus it is ok to use 32-bit integers for the node counts, because if the node count
 *    happens to be bigger than 2^32 for a node, this node will anyway be divided until the
 *    node count is smaller than the bucket size. We just have to make sure to prevent overflow,
 *    in MPI_Allreduce, therefore, maxCount should be set to 2^32/nRanks - 1 for distributed tree builds.
 */
template<class I, class Reduce = void>
std::tuple<std::vector<I>, std::vector<unsigned>>
computeOctree(const I* codesStart, const I* codesEnd, unsigned bucketSize,
              unsigned maxCount = std::numeric_limits<unsigned>::max(),
              std::vector<I>&& tree = std::vector<I>(0))
{
    if (!tree.size())
    {
        // tree containing just the root node
        tree.push_back(0);
        tree.push_back(nodeRange<I>(0));
    }

    std::vector<unsigned> counts(nNodes(tree));

    bool converged = false;
    while (!converged)
    {
        computeNodeCounts(tree.data(), counts.data(), nNodes(tree), codesStart, codesEnd, maxCount);
        if constexpr (!std::is_same_v<void, Reduce>) Reduce{}(counts);
        std::vector<I> balancedTree =
            rebalanceTree(tree.data(), counts.data(), nNodes(tree), bucketSize, &converged);

        swap(tree, balancedTree);
        counts.resize(nNodes(tree));
    }

    return std::make_tuple(std::move(tree), std::move(counts));
}

/*! \brief update the octree with a single rebalance/count step
 *
 * @tparam I              32- or 64-bit unsigned integer for morton code
 * @tparam Reduce         functor for global counts reduction in distributed builds
 * @param codesStart[in]  local particle Morton codes start
 * @param codesEnd[in]    local particle morton codes end
 * @param bucketSize[in]  maximum number of particles per node
 * @param tree[inout]     the octree leaf nodes (cornerstone format)
 * @param counts[inout]   the octree leaf node particle count
 * @param maxCount[in]    if actual node counts are higher, they will be capped to \a maxCount
 */
template<class I, class Reduce = void>
void updateOctree(const I* codesStart, const I* codesEnd, unsigned bucketSize,
                  std::vector<I>& tree, std::vector<unsigned>& counts,
                  unsigned maxCount = std::numeric_limits<unsigned>::max())
{
    std::vector<I> balancedTree = rebalanceTree(tree.data(), counts.data(), nNodes(tree), bucketSize);
    swap(tree, balancedTree);
    counts.resize(nNodes(tree));

    // local node counts
    computeNodeCounts(tree.data(), counts.data(), nNodes(tree), codesStart, codesEnd, maxCount);
    // global node count sums when using distributed builds
    if constexpr (!std::is_same_v<void, Reduce>) Reduce{}(counts);
}

/*! \brief Compute the halo radius of each node in the given octree
 *
 * This is the maximum distance beyond the node boundaries that a particle outside the
 * node could possibly interact with.
 *
 * TODO: Don't calculate the maximum smoothing length, calculate the maximum distance by
 *       which any of the particles plus radius protrude outside the node.
 *
 * \tparam Tin         float or double
 * \tparam Tout        float or double, usually float
 * \tparam I           32- or 64-bit unsigned integer type for Morton codes
 * \tparam IndexType   integer type for local particle array indices, 32-bit for fewer than 2^32 local particles
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
 * \param input        Radii per particle, i.e. the smoothing lengths in SPH, length = codesEnd - codesStart
 * \param output       Radius per node, length = @a nNodes
 */
template<class Tin, class Tout, class I, class IndexType>
void computeHaloRadii(const I* tree, int nNodes, const I* codesStart, const I* codesEnd,
                      const IndexType* ordering, const Tin* input, Tout* output)
{
    int firstNode = 0;
    int lastNode  = nNodes;
    if (codesStart != codesEnd)
    {
        firstNode = std::upper_bound(tree, tree + nNodes, *codesStart) - tree - 1;
        lastNode  = std::upper_bound(tree, tree + nNodes, *(codesEnd-1)) - tree;
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < firstNode; ++i)
        output[i] = 0;

    #pragma omp parallel for schedule(static)
    for (int i = lastNode; i < nNodes; ++i)
        output[i] = 0;

    #pragma omp parallel for
    for (int i = firstNode; i < lastNode; ++i)
    {
        I nodeStart = tree[i];
        I nodeEnd   = tree[i+1];

        // find elements belonging to particles in node i
        auto startIndex = IndexType(std::lower_bound(codesStart, codesEnd, nodeStart) - codesStart);
        auto endIndex   = IndexType(std::lower_bound(codesStart, codesEnd, nodeEnd)   - codesStart);

        Tin nodeMax = 0;
        for(IndexType p = startIndex; p < endIndex; ++p)
        {
            Tin nodeElement = input[ordering[p]];
            nodeMax       = std::max(nodeMax, nodeElement);
        }

        // note factor of 2 due to SPH conventions
        output[i] = Tout(2 * nodeMax);
    }
}

} // namespace cstone
