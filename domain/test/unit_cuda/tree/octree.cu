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
 * @brief internal octree GPU testing
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 */

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "gtest/gtest.h"

#include "cstone/tree/octree_gpu.h"
#include "cstone/tree/octree.hpp"
#include "cstone/tree/cs_util.hpp"

using namespace cstone;

template<class KeyType>
void compareAgainstCpu(const std::vector<KeyType>& tree)
{
    // upload cornerstone tree to device
    thrust::device_vector<KeyType> d_leaves = tree;

    OctreeData<KeyType, GpuTag> gpuTree;
    gpuTree.resize(nNodes(tree));

    buildOctreeGpu(rawPtr(d_leaves), gpuTree.data());

    Octree<KeyType> cpuTree;
    cpuTree.update(tree.data(), nNodes(tree));

    thrust::host_vector<KeyType> h_prefixes = gpuTree.prefixes;
    for (auto& p : h_prefixes)
    {
        p = decodePlaceholderBit(p);
    }

    EXPECT_EQ(cpuTree.numTreeNodes(), gpuTree.numLeafNodes + gpuTree.numInternalNodes);
    for (TreeNodeIndex i = 0; i < cpuTree.numTreeNodes(); ++i)
    {
        EXPECT_EQ(h_prefixes[i], cpuTree.codeStart(i));
    }

    thrust::host_vector<TreeNodeIndex> h_children = gpuTree.childOffsets;
    for (TreeNodeIndex i = 0; i < cpuTree.numTreeNodes(); ++i)
    {
        EXPECT_EQ(h_children[i], cpuTree.child(i, 0));
    }

    thrust::host_vector<TreeNodeIndex> h_parents = gpuTree.parents;
    for (TreeNodeIndex i = 0; i < cpuTree.numTreeNodes(); ++i)
    {
        EXPECT_EQ(h_parents[(i - 1) / 8], cpuTree.parent(i));
    }

    EXPECT_EQ(gpuTree.levelRange.size(), cpuTree.levelRange().size());
    EXPECT_EQ(gpuTree.levelRange.size(), maxTreeLevel<KeyType>{} + 2);
    thrust::host_vector<TreeNodeIndex> h_levelRange = gpuTree.levelRange;
    for (unsigned level = 0; level < gpuTree.levelRange.size(); ++level)
    {
        EXPECT_EQ(h_levelRange[level], cpuTree.levelRange()[level]);
    }
}

//! @brief This creates an irregular tree. Checks geometry relations between children and parents.
template<class KeyType>
void octreeIrregularL3()
{
    std::vector<KeyType> tree = OctreeMaker<KeyType>{}.divide().divide(0).divide(0, 2).divide(3).makeTree();

    compareAgainstCpu(tree);
}

TEST(OctreeGpu, irregularL3)
{
    octreeIrregularL3<unsigned>();
    octreeIrregularL3<uint64_t>();
}

template<class KeyType>
void octreeRegularL6()
{
    // uniform level-6 tree with 262144
    std::vector<KeyType> tree = makeUniformNLevelTree<KeyType>(64 * 64 * 64, 1);

    compareAgainstCpu(tree);
}

TEST(OctreeGpu, regularL6)
{
    octreeRegularL6<unsigned>();
    octreeRegularL6<uint64_t>();
}
