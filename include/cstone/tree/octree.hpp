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
 * @brief Generation of local and global octrees in cornerstone format
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * In the cornerstone format, the octree is stored as sequence of SFC codes
 * fulfilling three invariants. Each code in the sequence both signifies the
 * the start SFC code of an octree leaf node and serves as an upper SFC code bound
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
 *        the same parent) are present in the SFC code sequence
 *      - each node with index i is defined by its lowest possible SFC code at position
 *        i in the vector and the highest possible (excluding) SFC code at position i+1
 *        in the vector
 *      - a vector of length N represents N-1 leaf nodes
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
#include <tuple>

#include "cstone/sfc/common.hpp"
#include "cstone/primitives/scan.hpp"
#include "cstone/util/gsl-lite.hpp"

#include "definitions.h"

namespace cstone
{

//! @brief count particles in one tree node
template<class I>
CUDA_HOST_DEVICE_FUN
unsigned calculateNodeCount(const I* tree, TreeNodeIndex nodeIdx, const I* codesStart, const I* codesEnd, unsigned maxCount)
{
    I nodeStart = tree[nodeIdx];
    I nodeEnd   = tree[nodeIdx+1];

    // count particles in range
    auto rangeStart = stl::lower_bound(codesStart, codesEnd, nodeStart);
    auto rangeEnd   = stl::lower_bound(codesStart, codesEnd, nodeEnd);
    unsigned count  = rangeEnd - rangeStart;

    return stl::min(count, maxCount);
}

/*! @brief determine search bound for @p targetCode in an array of sorted particle SFC codes
 *
 * @tparam I          32- or 64-bit unsigned integer type
 * @param firstIdx    first (of two) search boundary, must be non-negative, but can exceed the codes range
 *                    (the guess for the location of @p targetCode in [codesStart:codesEnd]
 * @param targetCode  code in [codesStart:codesEnd] to look for
 * @param codesStart  particle SFC code array start
 * @param codesEnd    particle SFC code array end
 * @return            the sub-range in [codesStart:codesEnd] containing @p targetCode
 */
template<class I>
CUDA_HOST_DEVICE_FUN
pair<const I*> findSearchBounds(std::make_signed_t<I> firstIdx, I targetCode,
                                const I* codesStart, const I* codesEnd)
{
    using SI = std::make_signed_t<I>;
    SI nCodes = codesEnd - codesStart;

    // firstIdx must be an accessible index
    firstIdx = stl::min(nCodes-1, firstIdx);

    I firstCode = codesStart[firstIdx];
    if (firstCode == targetCode)
        firstIdx++;

    // d = 1 : search towards codesEnd
    // d = -1 : search towards codesStart
    SI d = (firstCode < targetCode) ? 1 : -1;

    SI targetCodeTimesD = targetCode * d;

    // determine search bound
    SI searchRange = 1;
    SI secondIndex = firstIdx + d;
    while(0 <= secondIndex && secondIndex < nCodes && d*codesStart[secondIndex] <= targetCodeTimesD)
    {
        searchRange *= 2;
        secondIndex = firstIdx + searchRange * d;
    }
    secondIndex = stl::max(SI(0), secondIndex);
    secondIndex = stl::min(nCodes, secondIndex);

    pair<const I*> searchBounds{codesStart + stl::min(firstIdx, secondIndex),
                                codesStart + stl::max(firstIdx, secondIndex)};
    return searchBounds;
}

/*! @brief calculate node counts with a guess to accelerate the binary search
 *
 * @tparam I                32- or 64-bit unsigned integer type
 * @param nodeIdx           the index of the node in @p tree to compute
 * @param tree              cornerstone octree
 * @param firstGuess        guess location of @p tree[nodeIdx] in [codesStart:codesEnd]
 * @param secondGuess       guess location of @p tree[nodeIdx+1] in [codesStart:codesEnd]
 * @param codesStart        particle SFC code array start
 * @param codesEnd          particle SFC code array end
 * @param maxCount          maximum particle count to report per node
 * @return                  the number of particles in the node at @p nodeIdx or maxCount,
 *                          whichever is smaller
 */
template<class I>
CUDA_HOST_DEVICE_FUN
unsigned updateNodeCount(TreeNodeIndex nodeIdx, const I* tree,
                         std::make_signed_t<I> firstGuess,
                         std::make_signed_t<I> secondGuess,
                         const I* codesStart, const I* codesEnd, unsigned maxCount)
{
    I nodeStart = tree[nodeIdx];
    I nodeEnd   = tree[nodeIdx+1];

    auto searchBounds   = findSearchBounds(firstGuess, nodeStart, codesStart, codesEnd);
    auto rangeStart     = stl::lower_bound(searchBounds[0], searchBounds[1], nodeStart);

    searchBounds  = findSearchBounds(secondGuess, nodeEnd, codesStart, codesEnd);
    auto rangeEnd = stl::lower_bound(searchBounds[0], searchBounds[1], nodeEnd);

    unsigned count = rangeEnd - rangeStart;
    return stl::min(count, maxCount);
}

/*! @brief count number of particles in each octree node
 *
 * @tparam I                  32- or 64-bit unsigned integer type
 * @param[in]    tree         octree nodes given as SFC codes of length @a nNodes+1
 *                            needs to satisfy the octree invariants
 * @param[inout] counts       output particle counts per node, length = @a nNodes
 * @param[in]    nNodes       number of nodes in tree
 * @param[in]    codesStart   sorted particle SFC code range start
 * @param[in]    codesEnd     sorted particle SFC code range end
 * @param[in]    maxCount     maximum particle count per node to store, this is used
 *                            to prevent overflow in MPI_Allreduce
 */
template<class I>
void computeNodeCounts(const I* tree, unsigned* counts, TreeNodeIndex nNodes, const I* codesStart, const I* codesEnd,
                       unsigned maxCount, bool useCountsAsGuess = false)
{
    TreeNodeIndex firstNode = 0;
    TreeNodeIndex lastNode  = nNodes;
    if (codesStart != codesEnd)
    {
        firstNode = std::upper_bound(tree, tree + nNodes, *codesStart) - tree - 1;
        lastNode  = std::upper_bound(tree, tree + nNodes, *(codesEnd-1)) - tree;
    }

    #pragma omp parallel for schedule(static)
    for (TreeNodeIndex i = 0; i < firstNode; ++i)
        counts[i] = 0;

    #pragma omp parallel for schedule(static)
    for (TreeNodeIndex i = lastNode; i < nNodes; ++i)
        counts[i] = 0;

    TreeNodeIndex nNonZeroNodes = lastNode - firstNode;
    const I* populatedTree = tree + firstNode;

    if (useCountsAsGuess)
    {
        exclusiveScan(counts + firstNode, nNonZeroNodes);
        #pragma omp parallel for schedule(static)
        for (TreeNodeIndex i = 0; i < nNonZeroNodes-1; ++i)
        {
            unsigned firstGuess   = counts[i + firstNode];
            unsigned secondGuess  = counts[i + firstNode + 1];
            counts[i + firstNode] = updateNodeCount(i, populatedTree, firstGuess, secondGuess,
                                                    codesStart, codesEnd, maxCount);
        }

        TreeNodeIndex lastIdx       = nNonZeroNodes-1;
        unsigned lastGuess          = counts[lastIdx + firstNode];
        counts[lastIdx + firstNode] = updateNodeCount(lastIdx, populatedTree, lastGuess, lastGuess,
                                                      codesStart, codesEnd, maxCount);
    }
    else
    {
        #pragma omp parallel for schedule(static)
        for (TreeNodeIndex i = 0; i < nNonZeroNodes; ++i)
        {
            counts[i + firstNode] = calculateNodeCount(populatedTree, i, codesStart, codesEnd, maxCount);
        }
    }
}

/*! @brief return the sibling index and level of the specified csTree node
 *
 * @tparam I         32- or 64-bit unsigned integer
 * @param csTree     cornerstone octree, length N
 * @param nodeIdx    node index in [0:N] of @p csTree to compute sibling index
 * @return           in first pair element: index in [0:8] if all 8 siblings of the specified
 *                   node are next to each other and at the same division level.
 *                   0 otherwise, i.e. if not all the 8 siblings exist in @p csTree
 *                   at the same division level
 *                   in second pair element: tree level of node at @p nodeIdx
 */
template<class I>
inline CUDA_HOST_DEVICE_FUN
pair<unsigned> siblingAndLevel(const I* csTree, TreeNodeIndex nodeIdx)
{
    I thisNode     = csTree[nodeIdx];
    I range        = csTree[nodeIdx + 1] - thisNode;
    unsigned level = treeLevel(range);
    unsigned siblingIdx = 0;
    if (level > 0)
    {
        siblingIdx = octalDigit(thisNode, level);
        bool siblings = (csTree[nodeIdx - siblingIdx + 8] == csTree[nodeIdx - siblingIdx] + nodeRange<I>(level - 1));
        if (!siblings) { siblingIdx = 0; }
    }

    return pair<unsigned>{siblingIdx, level};
}

//! @brief returns 0 for merging, 1 for no-change, 8 for splitting
template<class I>
CUDA_HOST_DEVICE_FUN
int calculateNodeOp(const I* tree, TreeNodeIndex nodeIdx, const unsigned* counts, unsigned bucketSize)
{
    auto p = siblingAndLevel(tree, nodeIdx);
    unsigned siblingIdx = p[0];
    unsigned level      = p[1];

    if (siblingIdx > 0) // 8 siblings next to each other, node can potentially be merged
    {
        // pointer to first node in sibling group
        auto g = counts + nodeIdx - siblingIdx;
        bool countMerge = (g[0]+g[1]+g[2]+g[3]+g[4]+g[5]+g[6]+g[7]) <= bucketSize;
        if (countMerge) { return 0; } // merge
    }

    if (counts[nodeIdx] > bucketSize && level < maxTreeLevel<I>{}) { return 8; } // split

    return 1; // default: do nothing
}

/*! @brief Compute split or fuse decision for each octree node in parallel
 *
 * @tparam I               32- or 64-bit unsigned integer type
 * @param[in] tree         octree nodes given as SFC codes of length @p nNodes
 *                         needs to satisfy the octree invariants
 * @param[in] counts       output particle counts per node, length = @p nNodes
 * @param[in] nNodes       number of nodes in tree
 * @param[in] bucketSize   maximum particle count per (leaf) node and
 *                         minimum particle count (strictly >) for (implicit) internal nodes
 * @param[out] nodeOps     stores rebalance decision result for each node, length = @p nNodes
 * @return                 true if converged, false
 *
 * For each node i in the tree, in nodeOps[i], stores
 *  - 0 if to be merged
 *  - 1 if unchanged,
 *  - 8 if to be split.
 */
template<class I, class LocalIndex>
bool rebalanceDecision(const I* tree, const unsigned* counts, TreeNodeIndex nNodes,
                       unsigned bucketSize, LocalIndex* nodeOps)
{
    bool converged = true;

    #pragma omp parallel
    {
        bool convergedThread = true;
        #pragma omp for
        for (TreeNodeIndex i = 0; i < nNodes; ++i)
        {
            int decision = calculateNodeOp(tree, i, counts, bucketSize);
            if (decision != 1) { convergedThread = false; }

            nodeOps[i] = decision;
        }
        if (!convergedThread) { converged = false; }
    }
    return converged;
}

/*! @brief transform old nodes into new nodes based on opcodes
 *
 * @tparam I          32- or 64-bit integer
 * @param nodeIndex   the node to process in @p oldTree
 * @param oldTree     the old tree
 * @param nodeOps     opcodes per old tree node
 * @param newTree     the new tree
 */
template<class I>
CUDA_HOST_DEVICE_FUN
void processNode(TreeNodeIndex nodeIndex, const I* oldTree, const TreeNodeIndex* nodeOps, I* newTree)
{
    I thisNode     = oldTree[nodeIndex];
    I range        = oldTree[nodeIndex+1] - thisNode;
    unsigned level = treeLevel(range);

    TreeNodeIndex opCode       = nodeOps[nodeIndex+1] - nodeOps[nodeIndex];
    TreeNodeIndex newNodeIndex = nodeOps[nodeIndex];

    if (opCode == 1)
    {
        newTree[newNodeIndex] = thisNode;
    }
    else if (opCode == 8)
    {
        for (int sibling = 0; sibling < 8; ++sibling)
        {
            newTree[newNodeIndex + sibling] = thisNode + sibling * nodeRange<I>(level + 1);
        }
    }
}

/*! @brief split or fuse octree nodes based on node counts relative to bucketSize
 *
 * @tparam I                  32- or 64-bit unsigned integer type
 * @param[in]    tree         cornerstone octree
 * @param[out]   newTree      rebalanced cornerstone octree
 * @param[in]    nodeOps      rebalance decision for each node, length @p numNodes(tree) + 1
 *                            will be overwritten
 */
template<class InputVector, class OutputVector>
void rebalanceTree(const InputVector& tree, OutputVector& newTree, TreeNodeIndex* nodeOps)
{
    using I = typename InputVector::value_type;
    TreeNodeIndex numNodes = nNodes(tree);

    exclusiveScan(nodeOps, numNodes + 1);
    newTree.resize(nodeOps[numNodes] + 1);

    #pragma omp parallel for schedule(static)
    for (TreeNodeIndex i = 0; i < numNodes; ++i)
    {
        processNode(i, tree.data(), nodeOps, newTree.data());
    }
    *newTree.rbegin() = nodeRange<I>(0);
}

/*! @brief update the octree with a single rebalance/count step
 *
 * @tparam I                 32- or 64-bit unsigned integer for SFC code
 * @tparam Reduce            functor for global counts reduction in distributed builds
 * @param[in]    codesStart  local particle SFC codes start
 * @param[in]    codesEnd    local particle SFC codes end
 * @param[in]    bucketSize  maximum number of particles per node
 * @param[inout] tree        the octree leaf nodes (cornerstone format)
 * @param[inout] counts      the octree leaf node particle count
 * @param[in]    maxCount    if actual node counts are higher, they will be capped to @p maxCount
 * @return                   true if tree was not modified, false otherwise
 */
template<class I, class Reduce = void>
bool updateOctree(const I* codesStart, const I* codesEnd, unsigned bucketSize,
                  std::vector<I>& tree, std::vector<unsigned>& counts,
                  unsigned maxCount = std::numeric_limits<unsigned>::max())
{
    std::vector<TreeNodeIndex> nodeOps(nNodes(tree) + 1);
    bool converged = rebalanceDecision(tree.data(), counts.data(), nNodes(tree), bucketSize, nodeOps.data());

    std::vector<I> newTree;
    rebalanceTree(tree, newTree, nodeOps.data());
    swap(tree, newTree);

    counts.resize(nNodes(tree));

    // local node counts
    computeNodeCounts(tree.data(), counts.data(), nNodes(tree), codesStart, codesEnd, maxCount, true);
    // global node count sums when using distributed builds
    if constexpr (!std::is_same_v<void, Reduce>) Reduce{}(counts);
    return converged;
}

/*! @brief compute an octree from SFC codes for a specified bucket size

 * @tparam I               32- or 64-bit unsigned integer type
 * @param[in] codesStart   particle SFC code sequence start
 * @param[in] codesEnd     particle SFC code sequence end
 * @param[in] bucketSize   maximum number of particles/codes per octree leaf node
 * @param[in] maxCount     if actual node counts are higher, they will be capped to @p maxCount
 * @return                 the tree and the node counts
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
              unsigned maxCount = std::numeric_limits<unsigned>::max())
{
    std::vector<I>        tree{0, nodeRange<I>(0)};
    std::vector<unsigned> counts{unsigned(codesEnd - codesStart)};

    bool converged = false;
    while (!converged)
    {
        converged = updateOctree<I, Reduce>(codesStart, codesEnd, bucketSize, tree, counts, maxCount);
    }

    return std::make_tuple(std::move(tree), std::move(counts));
}

/*! @brief create a cornerstone octree around a series of given SFC codes
 *
 * @tparam InputIterator  iterator to 32- or 64-bit unsigned integer
 * @param  firstCode      SFC code sequence start
 * @param  lastCode       SFC code sequence end
 * @return                the cornerstone octree containing all values in the given code sequence
 *                        plus any additional intermediate SFC codes between them required to fulfill
 *                        the cornerstone invariants.
 *
 * Typical application: Generation of a minimal global tree where the input code sequence
 * corresponds to the partitioning of the space filling curve into numMpiRanks intervals.
 *
 *  Requirements on the input sequence [firstCode:lastCode]:
 *      - must start with 0, i.e. *firstCode == 0
 *      - must end with 2^30 or 2^63, i.e. *(lastCode - 1) == nodeRange<CodeType>(0)
 *        with CodeType =- std::decay_t<decltype(*firstCode)>
 *      - must be sorted
 */
template<class InputIterator>
auto computeSpanningTree(InputIterator firstCode, InputIterator lastCode)
{
    using CodeType = std::decay_t<decltype(*firstCode)>;

    assert(lastCode - firstCode > 1);
    assert(*firstCode == 0 && *(lastCode-1) == nodeRange<CodeType>(0));

    TreeNodeIndex numIntervals = lastCode - firstCode - 1;

    std::vector<TreeNodeIndex> offsets(numIntervals + 1);
    for (TreeNodeIndex i = 0; i < numIntervals; ++i)
    {
        offsets[i] = spanSfcRange(firstCode[i], firstCode[i+1]);
    }

    exclusiveScanSerialInplace(offsets.data(), offsets.size(), 0);

    std::vector<CodeType> spanningTree(offsets.back() + 1);
    for (TreeNodeIndex i = 0; i < numIntervals; ++i)
    {
        spanSfcRange(firstCode[i], firstCode[i+1], spanningTree.data() + offsets[i]);
    }
    spanningTree.back() = nodeRange<CodeType>(0);

    return spanningTree;
}

/*! @brief Compute the halo radius of each node in the given octree
 *
 * This is the maximum distance beyond the node boundaries that a particle outside the
 * node could possibly interact with.
 *
 * TODO: Don't calculate the maximum smoothing length, calculate the maximum distance by
 *       which any of the particles plus radius protrude outside the node.
 *
 * @tparam Tin              float or double
 * @tparam Tout             float or double, usually float
 * @tparam I                32- or 64-bit unsigned integer type for SFC codes
 * @tparam IndexType        integer type for local particle array indices, 32-bit for fewer than 2^32 local particles
 * @param[in]  tree         octree nodes given as SFC codes of length @a nNodes+1
 *                          This function does not rely on octree invariants, sortedness of the nodes
 *                          is the only requirement.
 * @param[in]  nNodes       number of nodes in tree
 * @param[in]  codesStart   sorted SFC code range start of particles to count
 * @param[in]  codesEnd     sorted SFC code range end of particles to count
 * @param[in]  ordering     Access input according to @p ordering
 *                          The sequence input[ordering[i]], i=0,...,N must list the elements of input
 *                          (i.e. the smoothing lengths) such that input[i] is a property of the particle
 *                          (x[i], y[i], z[i]), with x,y,z sorted according to SFC ordering.
 * @param[in]  input        Radii per particle, i.e. the smoothing lengths in SPH, length = codesEnd - codesStart
 * @param[out] output       Radius per node, length = @a nNodes
 */
template<class Tin, class Tout, class I, class IndexType>
void computeHaloRadii(const I* tree, TreeNodeIndex nNodes, const I* codesStart, const I* codesEnd,
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
    for (TreeNodeIndex i = 0; i < firstNode; ++i)
        output[i] = 0;

    #pragma omp parallel for schedule(static)
    for (TreeNodeIndex i = lastNode; i < nNodes; ++i)
        output[i] = 0;

    #pragma omp parallel for
    for (TreeNodeIndex i = firstNode; i < lastNode; ++i)
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
            nodeMax         = std::max(nodeMax, nodeElement);
        }

        // note factor of 2 due to SPH conventions
        output[i] = Tout(2 * nodeMax);
    }
}

} // namespace cstone
