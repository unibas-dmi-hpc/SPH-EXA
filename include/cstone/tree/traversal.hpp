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
 * @brief binary tree traversal implementation
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/halos/boxoverlap.hpp"
#include "cstone/tree/octree_internal.hpp"

namespace cstone
{

template<class I>
inline bool overlapNode(const Octree<I>& octree, TreeNodeIndex nodeIndex, const IBox& collisionBox)
{
    return overlap(octree.codeStart(nodeIndex), 3 * octree.level(nodeIndex), collisionBox);
}

//constexpr int maxCoord = 1u<<maxTreeLevel<I>{};
//I iboxStart = imorton3D<I>(collisionBox.xmin(), collisionBox.ymin(), collisionBox.zmin());
//int xmax = collisionBox.xmax();
//int ymax = collisionBox.ymax();
//int zmax = collisionBox.zmax();
//if (xmax == maxCoord) xmax--;
//if (ymax == maxCoord) ymax--;
//if (zmax == maxCoord) zmax--;
//I iboxEnd   = imorton3D<I>(xmax, ymax, zmax);

//pair<I> commonBox = smallestCommonBox(iboxStart, iboxEnd);
//int iboxLevel = treeLevel<I>(commonBox[1] - commonBox[0]);

//for (int l = 1; l <= iboxLevel; ++l)
//{
//    int octant = octreeDigit(commonBox[0], l);
//    node = octree.child(node, octant);
//}

//if (octree.isLeaf(node))
//{
//    collisionList.add(node);
//    return;
//}

template <class I, class C, class A>
void traverse(const Octree<I>& octree, C&& continuationCriterion, A&& endpointAction)
{
    if (!continuationCriterion(0) || octree.isLeaf(0))
    {
        // root node is already the endpoint
        endpointAction(0);
        return;
    }

    TreeNodeIndex stack[64];
    stack[0] = 0;

    TreeNodeIndex stackPos = 1;
    TreeNodeIndex node     = 0; // start at the root

    TreeNodeIndex internalNodes = octree.nInternalNodes();
    do
    {
        for (int octant = 0; octant < 8; ++octant)
        {
            TreeNodeIndex child = octree.child(node, octant);
            bool descend = continuationCriterion(child);
            if (descend)
            {
                if (octree.isLeaf(child))
                {
                    endpointAction(child - internalNodes);
                }
                else
                {
                    assert (stackPos < 63);
                    stack[stackPos++] = child; // push
                }
            }
        }
        node = stack[--stackPos];

    } while (node != 0); // the root can only be obtained when the tree has been fully traversed
}

} // namespace cstone
