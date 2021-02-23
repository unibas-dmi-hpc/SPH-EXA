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
 * \brief  Compute the internal part of a cornerstone octree
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
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

#include "cstone/btree.hpp"
#include "cstone/octree.hpp"
#include "cstone/scan.hpp"

namespace cstone
{

//! \brief named bool to tag children of internal octree nodes
enum class ChildType : bool
{
    internal = true,
    leaf     = false
};

/*! \brief octree node for the internal part of cornerstone octrees
 *
 * @tparam I  32- or 64 bit unsigned integer
 */
template<class I>
struct OctreeNode
{

    /*! \brief the Morton code prefix
     *
     * Shared among all the node's children. Only the first prefixLength bits are relevant.
     */
    I   prefix;

    //! \brief octree division level, equals 1/3rd the number of bits in prefix to interpret
    int level;

    int btreeIndex;

    /*! \brief Child node indices
     *
     *  If childType[i] is ChildType::internal, children[i] is an internal node index,
     *  otherwise children[i] is the index of an octree leaf node.
     *  Note that in one case or the other, the index refers to two different arrays!
     */
    int       children[8];
    ChildType childTypes[8];
};

template<class I>
void createInternalOctree(const std::vector<BinaryNode<I>>& binaryTree, const std::vector<I>& tree)
{
    std::vector<unsigned> prefixes(binaryTree.size() + 1);

    for (int i = 0; i < binaryTree.size(); ++i)
    {
        int prefixLength = binaryTree[i].prefixLength;
        if (prefixLength%3 == 0)
            prefixes[i] = 1;
        else
            prefixes[i] = 0;
    }

    // stream compaction: scan and scatter
    exclusiveScan(prefixes.data(), prefixes.size());

    unsigned nInternalOctreeNodes = *prefixes.rbegin();
    std::vector<unsigned> scatterMap(nInternalOctreeNodes);

    // compaction step, scatterMap -> compacted list of binary nodes that correspond to octree nodes
    for (int i = 0; i < binaryTree.size(); ++i)
    {
        bool isOctreeNode = (prefixes[i+1] - prefixes[i]) == 1;
        if (isOctreeNode)
        {
            int octreeNodeIndex = prefixes[i];
            scatterMap[octreeNodeIndex] = i;
        }
    }

    std::vector<OctreeNode<I>> internalOctree(nInternalOctreeNodes);

    for (int i = 0; i < nInternalOctreeNodes; ++i)
    {
        OctreeNode<I> octreeNode;

        int bi = scatterMap[i]; // binary tree index
        octreeNode.prefix = binaryTree[bi].prefix;
        octreeNode.level  = binaryTree[bi].prefixLength / 3;
        octreeNode.btreeIndex = bi;

        if (binaryTree[bi].leftChild->leftChild->leftChild != nullptr) {
            octreeNode.children[0]   = binaryTree[bi].leftChild->leftChild->leftChild  - binaryTree.data();
            octreeNode.childTypes[0] = ChildType::internal;
        }
        else {
            octreeNode.children[0]   = binaryTree[bi].leftChild->leftChild->leftLeafIndex;
            octreeNode.childTypes[0] = ChildType::leaf;
        }

        if (binaryTree[bi].leftChild->leftChild->rightChild != nullptr) {
            octreeNode.children[1]   = binaryTree[bi].leftChild->leftChild->rightChild - binaryTree.data();
            octreeNode.childTypes[1] = ChildType::internal;
        }
        else{
            octreeNode.children[1]   = binaryTree[bi].leftChild->leftChild->rightLeafIndex;
            octreeNode.childTypes[1] = ChildType::leaf;
        }

        if (binaryTree[bi].leftChild->rightChild->leftChild != nullptr) {
            octreeNode.children[2]   = binaryTree[bi].leftChild->rightChild->leftChild - binaryTree.data();
            octreeNode.childTypes[2] = ChildType::internal;
        }
        else{
            octreeNode.children[2]   = binaryTree[bi].leftChild->rightChild->leftLeafIndex;
            octreeNode.childTypes[2] = ChildType::leaf;
        }

        internalOctree[i] = octreeNode;
    }

    for (int i = 0; i < binaryTree.size(); ++i)
        std::cout << std::setw(3) << prefixes[i] << " ";
    std::cout << std::endl;
    for (int i = 0; i < binaryTree.size(); ++i)
        std::cout << std::setw(3) << i << " ";
    std::cout << std::endl;

    std::cout << std::endl;

    for (int i = 0; i < internalOctree.size(); ++i)
        std::cout << std::setw(2) << internalOctree[i].btreeIndex << " ";
    std::cout << std::endl;
    for (int i = 0; i < internalOctree.size(); ++i)
        std::cout << std::setw(2) << internalOctree[i].level << " ";

    std::cout << std::endl;
    std::cout << std::endl;

    for (int i = 0; i < internalOctree.size(); ++i)
    {
        printf("octree node %d, prefix %10o, level %d, children ", i, internalOctree[i].prefix, internalOctree[i].level);
        for (int k = 0; k < 7; ++k)
        {
            if (internalOctree[i].childTypes[k] == ChildType::internal)
            {
                printf(" %10d ", internalOctree[i].children[k]);
            }
            else
            {
                printf(" %10o ", tree[internalOctree[i].children[k]]);
            }
        }
        std::cout << std::endl;
    }
}

} // namespace cstone
