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
                    assert (stackPos < 64);
                    stack[stackPos++] = child; // push
                }
            }
        }
        node = stack[--stackPos];

    } while (node != 0); // the root can only be obtained when the tree has been fully traversed
}

template <class I, class MAC, class M2L, class P2P>
void dualTraversal(const Octree<I>& octree, MAC&& continuation, M2L&& m2l, P2P&& p2p)
{
    using NodePair = pair<TreeNodeIndex>;

    if (octree.isLeaf(0)) { return; }

    NodePair stack[64];
    stack[0] = NodePair{0,0};

    int stackPos = 1;

    auto interact = [&octree, &continuation, &m2l, &p2p, &stackPos]
        (TreeNodeIndex a, TreeNodeIndex b, NodePair* stack_)
    {
        if (continuation(a, b))
        {
            if (octree.isLeaf(a) && octree.isLeaf(b)) { p2p(a, b); }
            else {
                assert(stackPos < 64);
                stack_[stackPos++] = NodePair{a, b};
            }
        }
        else { m2l(a, b); }
    };

    while (stackPos > 0)
    {
        NodePair nodePair = stack[--stackPos];
        TreeNodeIndex target = nodePair[0];
        TreeNodeIndex source = nodePair[1];

        if ((octree.level(target) < octree.level(source) && !octree.isLeaf(target))
            || octree.isLeaf(source))
        {
            int nChildren = (octree.isLeaf(target)) ? 0 : 8;
            for (int octant = 0; octant < nChildren; ++octant)
            {
                interact(octree.child(target, octant), source, stack);
            }
        }
        else
        {
            int nChildren = (octree.isLeaf(source)) ? 0 : 8;
            for (int octant = 0; octant < nChildren; ++octant)
            {
                interact(target, octree.child(source, octant), stack);
            }
        }
    }
}

} // namespace cstone
