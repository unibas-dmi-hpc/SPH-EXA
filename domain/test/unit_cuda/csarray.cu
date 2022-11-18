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
 * @brief Cornerstone octree GPU testing
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 */

#include "gtest/gtest.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "coord_samples/random.hpp"
#include "cstone/tree/csarray.hpp"
#include "cstone/tree/csarray_gpu.h"
#include "cstone/tree/cs_util.hpp"
#include "cstone/tree/update_gpu.cuh"

using namespace cstone;

TEST(CsArrayGpu, computeNodeCountsGpu)
{
    using KeyType = unsigned;

    // regular level-3 cornerstone tree with 512 leaves
    thrust::host_vector<KeyType> h_cstree = makeUniformNLevelTree<KeyType>(8 * 8 * 8, 1);
    // subdivide the first level-3 node
    for (int octant = 1; octant < 8; ++octant)
    {
        h_cstree.push_back(octant * nodeRange<KeyType>(4));
    }

    std::sort(begin(h_cstree), end(h_cstree));

    // create + upload tree to the device
    thrust::device_vector<KeyType> d_cstree = h_cstree;

    thrust::host_vector<KeyType> h_particleKeys;
    for (int nodeIdx = 1; nodeIdx < nNodes(h_cstree) - 1; ++nodeIdx)
    {
        // put 2 particles in each tree node, except the first and last node
        h_particleKeys.push_back(h_cstree[nodeIdx]);
        h_particleKeys.push_back(h_cstree[nodeIdx] + 1);
    }

    // upload particle codes to device
    thrust::device_vector<KeyType> d_particleKeys = h_particleKeys;

    thrust::device_vector<unsigned> d_counts(nNodes(d_cstree), 1);

    thrust::host_vector<unsigned> refCounts(nNodes(d_cstree), 2);
    // first and last nodes are empty
    refCounts[0]        = 0;
    *refCounts.rbegin() = 0;

    computeNodeCountsGpu(rawPtr(d_cstree), rawPtr(d_counts), nNodes(d_cstree), rawPtr(d_particleKeys),
                         rawPtr(d_particleKeys) + d_particleKeys.size(), std::numeric_limits<unsigned>::max(), false);
    thrust::host_vector<unsigned> h_counts = d_counts;
    EXPECT_EQ(h_counts, refCounts);

    // check again, using previous counts as guesses
    computeNodeCountsGpu(rawPtr(d_cstree), rawPtr(d_counts), nNodes(d_cstree), rawPtr(d_particleKeys),
                         rawPtr(d_particleKeys) + d_particleKeys.size(), std::numeric_limits<unsigned>::max(), true);
    h_counts = d_counts;
    EXPECT_EQ(h_counts, refCounts);
}

TEST(CsArrayGpu, rebalanceDecision)
{
    using KeyType       = unsigned;
    unsigned bucketSize = 8;

    thrust::device_vector<KeyType> tree = OctreeMaker<KeyType>{}.divide().divide(7).makeTree();
    thrust::device_vector<unsigned> counts(nNodes(tree), 1);
    counts[1] = 9;
    thrust::fill_n(counts.begin() + 8, 7, 0);

    thrust::device_vector<TreeNodeIndex> nodeOps(tree.size());
    computeNodeOpsGpu(rawPtr(tree), nNodes(tree), rawPtr(counts), bucketSize, rawPtr(nodeOps));

    // regular level-3 cornerstone tree with 512 leaves
    thrust::host_vector<TreeNodeIndex> h_nodeOps = nodeOps;

    thrust::host_vector<TreeNodeIndex> refNodeOps =
        std::vector<TreeNodeIndex>{0, 1, 9, 10, 11, 12, 13, 14, 15, 15, 15, 15, 15, 15, 15, 15};

    EXPECT_EQ(refNodeOps, h_nodeOps);
}

TEST(CsArrayGpu, rebalanceTree)
{
    using KeyType                       = unsigned;
    thrust::device_vector<KeyType> tree = OctreeMaker<KeyType>{}.divide().divide(7).makeTree();

    // node {1} to be split, nodes {7,i} are to be fused
    thrust::device_vector<TreeNodeIndex> nodeOps =
        std::vector<TreeNodeIndex>{0, 1, 9, 10, 11, 12, 13, 14, 15, 15, 15, 15, 15, 15, 15, 15};
    thrust::device_vector<KeyType> newTree(*nodeOps.rbegin() + 1);

    bool converged = rebalanceTreeGpu(rawPtr(tree), nNodes(tree), nNodes(newTree), rawPtr(nodeOps), rawPtr(newTree));

    // download tree from host
    thrust::host_vector<KeyType> h_tree    = newTree;
    thrust::host_vector<KeyType> reference = OctreeMaker<KeyType>{}.divide().divide(1).makeTree();
    EXPECT_EQ(h_tree, reference);
    EXPECT_FALSE(converged);
}

/*! @brief fixture for octree tests based on random particle distributions
 *
 * @tparam KeyType         32- or 64-bit unsigned integer
 *
 * These tests are already integration tests strictly speaking. They can be seen
 * as the second line of defense in case the unit tests above (with minimal and explict reference data)
 * fail to catch an error.
 */
template<class KeyType>
class OctreeFixtureGpu
{
public:
    OctreeFixtureGpu(unsigned numParticles, unsigned bucketSize)
    {
        d_codes = makeRandomGaussianKeys<KeyType>(numParticles);

        d_tree   = std::vector<KeyType>{0, nodeRange<KeyType>(0)};
        d_counts = std::vector<unsigned>{numParticles};

        thrust::device_vector<KeyType> tmpTree;
        thrust::device_vector<TreeNodeIndex> workArray;

        while (!updateOctreeGpu(thrust::raw_pointer_cast(d_codes.data()),
                                thrust::raw_pointer_cast(d_codes.data() + d_codes.size()), bucketSize, d_tree, d_counts,
                                tmpTree, workArray))
            ;
    }

    thrust::device_vector<KeyType> d_tree;
    thrust::device_vector<KeyType> d_codes;
    thrust::device_vector<unsigned> d_counts;
};

//! @brief build tree from random particles and compare against CPU
TEST(CsArrayGpu, computeOctreeRandom)
{
    using Integer = unsigned;

    int nParticles = 100000;
    int bucketSize = 64;

    // compute octree starting from default uniform octree
    auto particleKeys         = makeRandomGaussianKeys<Integer>(nParticles);
    auto [treeCpu, countsCpu] = computeOctree(particleKeys.data(), particleKeys.data() + nParticles, bucketSize);

    OctreeFixtureGpu<Integer> fixt(nParticles, bucketSize);

    // download tree from device
    thrust::host_vector<Integer> h_tree     = fixt.d_tree;
    thrust::host_vector<Integer> refTreeCpu = treeCpu;

    thrust::host_vector<Integer> h_counts     = fixt.d_counts;
    thrust::host_vector<Integer> refCountsCpu = countsCpu;

    EXPECT_EQ(h_tree, refTreeCpu);
    EXPECT_EQ(h_counts, refCountsCpu);
}

/*! @brief simulation of distributed tree
 *
 * In distributed octrees, the executing rank only has a part of the particle SFC codes, such that
 * many nodes in the tree are empty. Here this is simulated by removing a large connected part of the particle codes
 * and recomputing the node counts based on this subset of particle codes. The non-zero node counts should stay the
 * same.
 */
TEST(CsArrayGpu, distributedMockUp)
{
    using CodeType = unsigned;

    int nParticles = 100000;
    int bucketSize = 64;

    OctreeFixtureGpu<CodeType> fixt(nParticles, bucketSize);

    thrust::device_vector<CodeType> d_counts_orig = fixt.d_counts;

    // omit first and last tenth of nodes
    TreeNodeIndex Nodes     = nNodes(fixt.d_tree);
    TreeNodeIndex firstNode = Nodes / 10;
    TreeNodeIndex lastNode  = Nodes - Nodes / 10;

    // determine the part of the tree that will be empty
    thrust::host_vector<CodeType> h_codes = fixt.d_codes;
    unsigned firstParticleIdx =
        stl::lower_bound(h_codes.begin(), h_codes.end(), fixt.d_tree[firstNode]) - h_codes.begin();
    unsigned lastParticleIdx =
        stl::lower_bound(h_codes.begin(), h_codes.end(), fixt.d_tree[lastNode]) - h_codes.begin();
    std::cout << firstNode << " " << lastNode << std::endl;
    std::cout << firstParticleIdx << " " << lastParticleIdx << std::endl;

    bool useCountsAsGuess = true;
    computeNodeCountsGpu(thrust::raw_pointer_cast(fixt.d_tree.data()), thrust::raw_pointer_cast(fixt.d_counts.data()),
                         nNodes(fixt.d_tree), thrust::raw_pointer_cast(fixt.d_codes.data() + firstParticleIdx),
                         thrust::raw_pointer_cast(fixt.d_codes.data() + lastParticleIdx),
                         std::numeric_limits<unsigned>::max(), useCountsAsGuess);

    thrust::device_vector<CodeType> d_counts_ref = d_counts_orig;
    thrust::fill(d_counts_ref.begin(), d_counts_ref.begin() + firstNode, 0);
    thrust::fill(d_counts_ref.begin() + lastNode, d_counts_ref.end(), 0);

    EXPECT_EQ(fixt.d_counts, d_counts_ref);
}
