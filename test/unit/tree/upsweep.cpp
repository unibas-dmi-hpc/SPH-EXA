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
 * \brief  Octree upsweep tests
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 */

#include "gtest/gtest.h"

#include "cstone/tree/upsweep.hpp"
#include "cstone/tree/octree_util.hpp"

using namespace cstone;

TEST(Upsweep, sum)
{
    using I = unsigned;

    std::vector<I> tree = OctreeMaker<I>{}.divide().divide(0).divide(0,2).divide(3).makeTree();

    Octree<I> fullTree;
    fullTree.update(tree.data(), tree.data() + tree.size());
    EXPECT_EQ(fullTree.nTreeNodes(), 33);
    EXPECT_EQ(fullTree.nLeaves(), 29);
    EXPECT_EQ(fullTree.nInternalNodes(), 4);

    //checkConnectivity(fullTree);

    for (int i = 0; i < fullTree.nTreeNodes(); ++i)
    {
        printf("node %3d, prefix %10o, level %1d\n", i, fullTree.codeStart(i), fullTree.level(i));
    }
}
