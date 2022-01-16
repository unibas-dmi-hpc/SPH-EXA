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
 * @brief Test halo radii functionality
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "cstone/tree/octree.hpp"
#include "cstone/halos/radii.hpp"

using namespace cstone;

TEST(CornerstoneOctree, computeHaloRadii)
{
    using KeyType = unsigned;

    std::vector<KeyType> tree{0, 8, 16, 24, 32};

    std::vector<KeyType> particleCodes{0, 4, 8, 14, 20, 24, 25, 26, 31};
    std::vector<float> smoothingLs{2, 1, 4, 3, 5, 8, 2, 1, 3};
    std::vector<float> hMaxPerNode{4, 8, 10, 16};

    std::vector<float> probe(hMaxPerNode.size());

    std::vector<LocalParticleIndex> ordering(particleCodes.size());

    computeHaloRadii<KeyType>(tree.data(), nNodes(tree), particleCodes, smoothingLs.data(), probe.data());

    EXPECT_EQ(probe, hMaxPerNode);
}

template<class KeyType>
void computeHaloRadiiSTree()
{
    std::vector<KeyType> cornerstones{0, 1, nodeRange<KeyType>(0) - 1, nodeRange<KeyType>(0)};
    std::vector<KeyType> tree = computeSpanningTree<KeyType>(cornerstones);

    /// 2 particles in the first and last node
    std::vector<KeyType> particleCodes{0, 0, nodeRange<KeyType>(0) - 1, nodeRange<KeyType>(0) - 1};

    std::vector<double> smoothingLengths{0.21, 0.2, 0.2, 0.22};

    std::vector<LocalParticleIndex> ordering(particleCodes.size());

    std::vector<double> haloRadii(nNodes(tree), 0);
    computeHaloRadii<KeyType>(tree.data(), nNodes(tree), particleCodes, smoothingLengths.data(), haloRadii.data());

    std::vector<double> referenceHaloRadii(nNodes(tree));
    referenceHaloRadii.front() = 0.42;
    referenceHaloRadii.back() = 0.44;

    EXPECT_EQ(referenceHaloRadii, haloRadii);
}

TEST(CornerstoneOctree, computeHaloRadii_spanningTree)
{
    computeHaloRadiiSTree<unsigned>();
    computeHaloRadiiSTree<uint64_t>();
}
