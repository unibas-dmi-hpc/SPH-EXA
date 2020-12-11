#pragma once

/*! \brief \file binary radix tree implementation
 *
 * Algorithm published in https://dl.acm.org/doi/10.5555/2383795.2383801
 * and further illustrated at
 * https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
 *
 * Most SFC-based functionality for SPH like domain decomposition, global octree build
 * and neighbor search does not need tree-traversal and explicit construction
 * of tree-internal nodes.
 *
 * The halo search problem is an exception. Finding all nodes in the global octree
 * that overlap with a given node that is enlarged by the search radius in all
 * dimensions is essentially the same as collision detection between objects
 * in 3D graphics.
 *
 * The best way to implement it is by constructing the internal tree part over the
 * global octree leaf nodes as a binary radix tree. Each internal node of the binary
 * radix tree can be constructed independently and thus the algorithm is ideally suited
 * for GPUs. Subsequent tree traversal to detect collisions can also be done for all leaf
 * nodes in parallel.
 */

#include "sfc/clz.hpp"
#include "sfc/mortoncode.hpp"

namespace sphexa {

/*! \brief binary radix tree node
 *
 * @tparam I 32 or 64 bit unsigned integer
 *
 * (final content TBD)
 */
template<class I>
struct BinaryNode
{
    BinaryNode* leftChild;
    BinaryNode* rightChild;
    BinaryNode* parent;

    I        prefix;
    unsigned prefixLength;
    I        mortonCode;
};

/*! \brief calculate common prefix (cpr) of two morton keys
 *
 * @tparam I    32 or 64 bit unsigned integer
 * @param key1  first morton code key
 * @param key2  second morton code key
 * @return      number of identical bits, counting from MSB
 *
 */
template<class I>
I cpr(I key1, I key2)
{
    return countLeadingZeros(key1 ^ key2);
}

/*! \brief find position of first differing bit
 *
 * @tparam I                 32 or 64 bit unsigned integer
 * @param sortedMortonCodes
 * @param first              first range index
 * @param last               last rang index
 * @return                   position of morton
 */
template<class I>
int findSplit(I*  sortedMortonCodes,
              int first,
              int last)
{
    // Identical Morton codes => split the range in the middle.
    I firstCode = sortedMortonCodes[first];
    I lastCode  = sortedMortonCodes[last];

    if (firstCode == lastCode)
        return (first + last) >> 1;

    // Calculate the number of highest bits that are the same
    // for all objects, using the count-leading-zeros intrinsic.
    int commonPrefix = cpr(firstCode, lastCode);

    // Use binary search to find where the next bit differs.
    // Specifically, we are looking for the highest object that
    // shares more than commonPrefix bits with the first one.
    int split = first; // initial guess
    int step = last - first;

    do
    {
        step = (step + 1) >> 1;      // exponential decrease
        int newSplit = split + step; // proposed new position

        if (newSplit < last)
        {
            I splitCode = sortedMortonCodes[newSplit];
            int splitPrefix = cpr(firstCode, splitCode);
            if (splitPrefix > commonPrefix)
                split = newSplit; // accept proposal
        }
    }
    while (step > 1);

    return split;
}

/*! \brief construct the internal binary tree node with index idx
 *
 * @tparam I
 * @param codes
 * @param leaves
 * @param nLeaves
 * @param internalNodes
 * @param idx
 */
template<class I>
void constructInternalNode(I* codes, BinaryNode<I>* leaves, int nLeaves, BinaryNode<I>* internalNodes, int idx)
{
    int d = 1;
    int minPrefixLength = unusedBits<I>{} - 1;

    if (idx > 0)
    {
        d = (cpr(codes[idx], codes[idx + 1]) > cpr(codes[idx], codes[idx - 1])) ? 1 : -1;
        minPrefixLength = cpr(codes[idx], codes[idx-d]);
    }

    //int jdx = idx + d;
    //while (0 < jdx && jdx < nLeaves - 1
    //       && cpr(codes[jdx+d], codes[idx]) > minPrefixLength)
    //{
    //    jdx += d;
    //}

    // find max search range
    int jSearchRange = 2;
    int upperJ       = idx + jSearchRange * d;
    while(0 <= upperJ && upperJ < nLeaves
          && cpr(codes[idx], codes[upperJ]) > minPrefixLength)
    {
        jSearchRange *= 2;
        upperJ = idx + jSearchRange * d;
    }

    // prune max search range if out of bounds
    if (upperJ >= nLeaves)
    {
        jSearchRange = nLeaves - idx;
    }
    if (upperJ < 0)
    {
        jSearchRange = idx;
    }

    // binary search to determine the second node range index jdx
    int nodeLength = 0;
    int step = jSearchRange;
    do
    {
        step = (step + 1) / 2;
        if (cpr(codes[idx], codes[idx + (nodeLength+step)*d]) > minPrefixLength)
        {
            nodeLength += step;
        }

    } while (step > 1);

    int jdx = idx + nodeLength * d;

    // find position of highest differing bit between [idx, jdx]
    int gamma = findSplit(codes, std::min(jdx, idx), std::max(jdx, idx));

    // establish child relationships
    if (std::min(jdx, idx) == gamma)
    {
        internalNodes[idx].leftChild = leaves + gamma;
    }
    else
    {
        internalNodes[idx].leftChild = internalNodes + gamma;
    }

    if (std::max(jdx,idx) == gamma + 1)
    {
        internalNodes[idx].rightChild = leaves + gamma + 1;
    }
    else
    {
        internalNodes[idx].rightChild = internalNodes + gamma + 1;
    }
}

} // namespace sphexa
