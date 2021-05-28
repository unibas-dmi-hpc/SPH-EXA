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
 * @brief Generic octree traversal tests
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "cstone/tree/octree_internal.hpp"
#include "cstone/tree/octree_essential.hpp"
#include "cstone/tree/octree_util.hpp"
#include "cstone/tree/traversal.hpp"

namespace cstone
{

template<class I>
void surfaceDetection()
{
    std::vector<I> tree = makeUniformNLevelTree<I>(64, 1);

    Octree<I> fullTree;
    fullTree.update(tree.data(), tree.data() + tree.size());

    IBox targetBox = makeIBox(tree[1], tree[2]);

    std::vector<IBox> treeBoxes(fullTree.nTreeNodes());
    for (TreeNodeIndex i = 0; i < fullTree.nTreeNodes(); ++i)
    {
        treeBoxes[i] = makeIBox(fullTree.codeStart(i), fullTree.codeEnd(i));
    }

    auto isSurface = [targetBox, bbox = Box<double>(0,1), boxes=treeBoxes.data()](TreeNodeIndex idx)
    {
        double distance = minDistanceSq<double, I>(targetBox, boxes[idx], bbox);
        return distance == 0.0;
    };

    std::vector<TreeNodeIndex> surfaceNodes;
    auto saveIndex = [&surfaceNodes](TreeNodeIndex idx) { surfaceNodes.push_back(idx); };

    singleTraversal(fullTree, isSurface, saveIndex);

    std::sort(begin(surfaceNodes), end(surfaceNodes));
    std::vector<TreeNodeIndex> reference{0,1,2,3,4,5,6,7,8,10,12,14};
    EXPECT_EQ(surfaceNodes, reference);
}

TEST(Traversal, surfaceDetection)
{
    surfaceDetection<unsigned>();
    surfaceDetection<uint64_t>();
}

//! @brief mac criterion refines all nodes, traverses the entire tree and finds all leaf-pairs
template<class I>
void dualTraversalAllPairs()
{
    Octree<I> fullTree;
    fullTree.update(OctreeMaker<I>{}.divide().divide(0).divide(0,7).makeTree());

    std::vector<pair<TreeNodeIndex>> pairs;

    auto allPairs = [](TreeNodeIndex a, TreeNodeIndex b) { return true; };

    auto m2l = [](TreeNodeIndex a, TreeNodeIndex b) { };
    auto p2p = [&pairs](TreeNodeIndex a, TreeNodeIndex b) { pairs.emplace_back(a, b); };

    dualTraversal(fullTree, 0, 0, allPairs, m2l, p2p);

    std::sort(begin(pairs), end(pairs));
    auto uit = std::unique(begin(pairs), end(pairs));
    EXPECT_EQ(uit, end(pairs));
    EXPECT_EQ(pairs.size(), 484); // 22 leaves ^2 = 484
}

TEST(Traversal, dualTraversalAllPairs)
{
    dualTraversalAllPairs<unsigned>();
    dualTraversalAllPairs<uint64_t>();
}

/*! @brief dual traversal with A, B across a focus range and touching each other
 *
 * This finds all pairs of leaves (a,b) that touch each other and with
 * a inside the focus and b outside.
 */
template<class I>
void dualTraversalNeighbors()
{
    Octree<I> octree;
    octree.update(makeUniformNLevelTree<I>(64, 1));

    Box<float> box(0,1);

    I focusStart = octree.codeStart(octree.toInternal(0));
    I focusEnd   = octree.codeStart(octree.toInternal(8));

    auto crossFocusSurfacePairs = [focusStart, focusEnd, &tree = octree, &box]
        (TreeNodeIndex a, TreeNodeIndex b)
    {
        bool aFocusOverlap = overlapTwoRanges(focusStart, focusEnd, tree.codeStart(a), tree.codeEnd(a));
        bool bInFocus      = containedIn(tree.codeStart(b), tree.codeEnd(b), focusStart, focusEnd);
        if (!aFocusOverlap || bInFocus) { return false; }

        IBox aBox = makeIBox(tree.codeStart(a), tree.codeEnd(a));
        IBox bBox = makeIBox(tree.codeStart(b), tree.codeEnd(b));
        return minDistanceSq<float, I>(aBox, bBox, box) == 0.0;
    };

    std::vector<pair<TreeNodeIndex>> pairs;
    auto p2p = [&pairs](TreeNodeIndex a, TreeNodeIndex b) { pairs.emplace_back(a, b); };

    auto m2l = [](TreeNodeIndex a, TreeNodeIndex b) {};

    dualTraversal(octree, 0, 0, crossFocusSurfacePairs, m2l, p2p);

    EXPECT_EQ(pairs.size(), 61);
    std::sort(begin(pairs), end(pairs));
    for(auto p : pairs)
    {
        auto a = p[0];
        auto b = p[1];
        //std::cout << a - octree.nInternalNodes() << " " << b - octree.nInternalNodes() << std::endl;
        // a in focus
        EXPECT_TRUE(octree.codeStart(a) >= focusStart && octree.codeEnd(a) <= focusEnd);
        // b outside focus
        EXPECT_TRUE(octree.codeStart(b) >= focusEnd || octree.codeEnd(a) <= focusStart);
        // a and be touch each other
        IBox aBox = makeIBox(octree.codeStart(a), octree.codeEnd(a));
        IBox bBox = makeIBox(octree.codeStart(b), octree.codeEnd(b));
        EXPECT_FLOAT_EQ((minDistanceSq<float, I>(aBox, bBox, box)), 0.0);
    }
}

TEST(Traversal, dualTraversalNeighbors)
{
    dualTraversalNeighbors<unsigned>();
    dualTraversalNeighbors<uint64_t>();
}

} // namespace cstone