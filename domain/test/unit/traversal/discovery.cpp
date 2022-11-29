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
 * @brief Halo discovery tests
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "cstone/traversal/collisions.hpp"
#include "cstone/tree/cs_util.hpp"

#include "collisions_a2a.hpp"

using namespace cstone;

template<class KeyType, class T>
std::vector<int> findHalosAll2All(gsl::span<const KeyType> tree,
                                  const std::vector<T>& haloRadii,
                                  const Box<T>& box,
                                  TreeNodeIndex firstNode,
                                  TreeNodeIndex lastNode)
{
    std::vector<int> flags(nNodes(tree));
    auto collisions = findCollisionsAll2all(tree, haloRadii, box);

    for (TreeNodeIndex i = firstNode; i < lastNode; ++i)
    {
        for (TreeNodeIndex cidx : collisions[i])
        {
            if (cidx < firstNode || cidx >= lastNode) { flags[cidx] = 1; }
        }
    }

    return flags;
}

template<class KeyType>
void findHalosFlags()
{
    std::vector<KeyType> tree = makeUniformNLevelTree<KeyType>(64, 1);

    Box<double> box(0, 1);

    // size of one node is 0.25^3
    std::vector<double> interactionRadii(nNodes(tree), 0.1);

    Octree<KeyType> octree;
    octree.update(tree.data(), nNodes(tree));

    {
        std::vector<int> collisionFlags(nNodes(tree), 0);
        findHalos(octree.nodeKeys().data(), octree.childOffsets().data(), octree.toLeafOrder().data(), tree.data(),
                  interactionRadii.data(), box, 0, 32, collisionFlags.data());

        std::vector<int> reference = findHalosAll2All<KeyType>(tree, interactionRadii, box, 0, 32);

        // consistency check: the surface of the first 32 nodes with the last 32 nodes is 16 nodes
        EXPECT_EQ(16, std::accumulate(collisionFlags.begin(), collisionFlags.end(), 0));
        EXPECT_EQ(collisionFlags, reference);
    }
    {
        std::vector<int> collisionFlags(nNodes(tree), 0);
        findHalos(octree.nodeKeys().data(), octree.childOffsets().data(), octree.toLeafOrder().data(), tree.data(),
                  interactionRadii.data(), box, 32, 64, collisionFlags.data());

        std::vector<int> reference = findHalosAll2All<KeyType>(tree, interactionRadii, box, 32, 64);

        // consistency check: the surface of the first 32 nodes with the last 32 nodes is 16 nodes
        EXPECT_EQ(16, std::accumulate(collisionFlags.begin(), collisionFlags.end(), 0));
        EXPECT_EQ(collisionFlags, reference);
    }
}

TEST(HaloDiscovery, findHalosFlags)
{
    findHalosFlags<unsigned>();
    findHalosFlags<uint64_t>();
}
