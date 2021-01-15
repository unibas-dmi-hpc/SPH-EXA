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
 * \brief Halo discovery tests
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "sfc/halodiscovery.hpp"

using namespace sphexa;

/*! \brief find halo test
 *
 * A regular 4x4x4 tree with 64 nodes is assigned to 2 ranks,
 * such that nodes 0-32 are on rank 0 and 32-64 on rank 1,
 * or, in x,y,z coordinates,
 *
 * nodes (0-2, 0-4, 0-4) -> rank 0
 * nodes (2-4, 0-4, 0-4) -> rank 1
 *
 * Halo search radius is less then a node edge length, so the halo nodes are
 *
 * (2, 0-4, 0-4) halos of rank 0
 * (1, 0-4, 0-4) halos of rank 1
 */
template <class I>
void findHalos()
{
    // a tree with 4 subdivisions along each dimension, 64 nodes
    std::vector<I> tree = makeUniformNLevelTree<I>(64, 1);

    // two domains
    SpaceCurveAssignment<I> assignment(2);
    assignment.addRange(Rank(0), tree[0], tree[32], 64);
    assignment.addRange(Rank(1), tree[32], tree[64], 64);

    Box<double> box(0, 1);

    // size of one node is 0.25^3
    std::vector<double> interactionRadii(nNodes(tree), 0.1);

    std::vector<pair<int>> refPairs0;
    for (int i = 0; i < nNodes(tree) / 2; ++i)
        for (int j = nNodes(tree) / 2; j < nNodes(tree); ++j)
        {
            if (overlap(tree[i], tree[i + 1], makeHaloBox(tree[j], tree[j + 1], interactionRadii[j], box)))
            {
                refPairs0.emplace_back(i, j);
            }
        }
    std::sort(begin(refPairs0), end(refPairs0));
    EXPECT_EQ(refPairs0.size(), 100);

    {
        std::vector<pair<int>> testPairs0;
        findHalos(tree, interactionRadii, box, assignment, 0, testPairs0);
        std::sort(begin(testPairs0), end(testPairs0));

        EXPECT_EQ(testPairs0.size(), 100);
        EXPECT_EQ(testPairs0, refPairs0);
    }

    auto refPairs1 = refPairs0;
    for (auto& p : refPairs1)
        std::swap(p[0], p[1]);
    std::sort(begin(refPairs1), end(refPairs1));

    {
        std::vector<pair<int>> testPairs1;
        findHalos(tree, interactionRadii, box, assignment, 1, testPairs1);
        std::sort(begin(testPairs1), end(testPairs1));
        EXPECT_EQ(testPairs1.size(), 100);
        EXPECT_EQ(testPairs1, refPairs1);
    }
}

TEST(HaloDiscovery, findHalos)
{
    findHalos<unsigned>();
    findHalos<uint64_t>();
}


//! \brief test processing of halo pair nodes into send/receive node lists
template <class I>
void computeSendRecvNodeList()
{
    // a tree with 4 subdivisions along each dimension, 64 nodes
    std::vector<I> tree = makeUniformNLevelTree<I>(64, 1);

    // two domains
    SpaceCurveAssignment<I> assignment(2);
    assignment.addRange(Rank(0), tree[0], tree[32], 64);
    assignment.addRange(Rank(1), tree[32], tree[64], 64);

    Box<double> box(0, 1);

    // size of one node is 0.25^3
    std::vector<double> interactionRadii(nNodes(tree), 0.1);

    std::vector<pair<int>> haloPairs;
    for (int i = 0; i < nNodes(tree) / 2; ++i)
        for (int j = nNodes(tree) / 2; j < nNodes(tree); ++j)
        {
            if (overlap(tree[i], tree[i + 1], makeHaloBox(tree[j], tree[j + 1], interactionRadii[j], box)))
            {
                haloPairs.emplace_back(i, j);
            }
        }
    std::sort(begin(haloPairs), end(haloPairs));

    {
        std::vector<std::vector<int>> incomingHalos;
        std::vector<std::vector<int>> outgoingHalos;

        computeSendRecvNodeList<I>(tree, assignment, haloPairs, incomingHalos, outgoingHalos);

        std::vector<std::vector<int>> refIncomingHalos(assignment.nRanks());
        std::vector<std::vector<int>> refOutgoingHalos(assignment.nRanks());

        std::vector<int> frontier0{4,5,6,7,12,13,14,15,20,21,22,23,28,29,30,31};
        std::vector<int> frontier1{32,33,34,35,40,41,42,43,48,49,50,51,56,57,58,59};

        refIncomingHalos[1] = frontier1;
        refOutgoingHalos[1] = frontier0;

        EXPECT_EQ(incomingHalos, refIncomingHalos);
        EXPECT_EQ(outgoingHalos, refOutgoingHalos);
    }
}

TEST(HaloDiscovery, sendRecvNodeList)
{
    computeSendRecvNodeList<unsigned>();
    computeSendRecvNodeList<uint64_t>();
}