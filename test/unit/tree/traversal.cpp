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

    traverse(fullTree, isSurface, saveIndex);

    std::sort(begin(surfaceNodes), end(surfaceNodes));
    std::vector<TreeNodeIndex> reference{0,1,2,3,4,5,6,7,8,10,12,14};
    EXPECT_EQ(surfaceNodes, reference);
}

TEST(Traversal, surfaceDetection)
{
    surfaceDetection<unsigned>();
    surfaceDetection<uint64_t>();
}

} // namespace cstone