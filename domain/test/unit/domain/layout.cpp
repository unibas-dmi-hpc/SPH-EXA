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
 * @brief Test functions used to determine the arrangement of halo and assigned particles
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "cstone/domain/layout.hpp"
#include "cstone/tree/octree_util.hpp"

using namespace cstone;

//! @brief test processing of halo pair nodes into send/receive node lists
TEST(Layout, sendRecvNodeList)
{
    // two domains
    SpaceCurveAssignment assignment(2);
    assignment.addRange(Rank(0), 0, 10, 0);
    assignment.addRange(Rank(1), 10, 20, 0);

    std::vector<pair<TreeNodeIndex>> haloPairs{{9, 10}, {9, 11}, {8, 11}};

    std::vector<std::vector<TreeNodeIndex>> incomingHalos;
    std::vector<std::vector<TreeNodeIndex>> outgoingHalos;
    computeSendRecvNodeList(assignment, haloPairs, incomingHalos, outgoingHalos);

    std::vector<std::vector<TreeNodeIndex>> refIncomingHalos(assignment.numRanks());
    std::vector<std::vector<TreeNodeIndex>> refOutgoingHalos(assignment.numRanks());

    std::vector<TreeNodeIndex> frontier0{8, 9};
    std::vector<TreeNodeIndex> frontier1{10, 11};

    refIncomingHalos[1] = frontier1;
    refOutgoingHalos[1] = frontier0;

    EXPECT_EQ(incomingHalos, refIncomingHalos);
    EXPECT_EQ(outgoingHalos, refOutgoingHalos);
}

TEST(Layout, flattenNodeList)
{
    std::vector<std::vector<TreeNodeIndex>> grouped{{0, 1, 2}, {3, 4, 5}, {6}, {}};

    std::vector<TreeNodeIndex> flattened = flattenNodeList(grouped);

    std::vector<TreeNodeIndex> ref{0, 1, 2, 3, 4, 5, 6};
    EXPECT_EQ(flattened, ref);
}

TEST(Layout, computeLayoutOffsets)
{
    std::vector<TreeNodeIndex> localNodes{4, 10};
    std::vector<TreeNodeIndex> halos{1, 3, 14, 15, 16, 21, 30};

    TreeNodeIndex nNodes = 32;
    std::vector<unsigned> nodeCounts(nNodes, 1);

    nodeCounts[1] = 2;
    nodeCounts[3] = 3;
    nodeCounts[4] = 5;
    nodeCounts[16] = 6;
    nodeCounts[24] = 8;
    nodeCounts[30] = 9;

    std::vector<TreeNodeIndex> presentNodes, offsets;
    computeLayoutOffsets(localNodes[0], localNodes[1], halos, nodeCounts, presentNodes, offsets);

    std::vector<int> refPresentNodes{1, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 21, 30};
    // counts                        2,3,5,1,1,1,1,1,1, 1, 6, 1, 9
    EXPECT_EQ(presentNodes, refPresentNodes);

    std::vector<int> refOffsets{0, 2, 5, 10, 11, 12, 13, 14, 15, 16, 17, 23, 24, 33};
    EXPECT_EQ(offsets, refOffsets);
}

TEST(Layout, createHaloExchangeList)
{
    int nRanks = 3;
    std::vector<std::vector<TreeNodeIndex>> outgoingHaloNodes(nRanks);
    outgoingHaloNodes[1].push_back(1);
    outgoingHaloNodes[1].push_back(10);
    outgoingHaloNodes[2].push_back(12);
    outgoingHaloNodes[2].push_back(22);

    std::vector<TreeNodeIndex> presentNodes{1, 2, 10, 12, 20, 22};
    std::vector<int> offsets{0, 1, 3, 6, 10, 15, 21};

    SendList sendList = createHaloExchangeList(outgoingHaloNodes, presentNodes, offsets);

    SendList refSendList(nRanks);
    refSendList[1].addRange(0, 1);
    refSendList[1].addRange(3, 6);
    refSendList[2].addRange(6, 10);
    refSendList[2].addRange(15, 21);

    EXPECT_EQ(sendList, refSendList);
}

TEST(Layout, computeHaloReceiveList)
{
    std::vector<LocalParticleIndex> layout{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<int> haloFlags{1, 0, 1, 1, 0, 0, 0, 1, 1, 0};

    std::vector<int> peers{0, 2};

    int numRanks = 3;
    std::vector<TreeIndexPair> assignment(numRanks);

    assignment[0] = TreeIndexPair(0, 4);
    assignment[1] = TreeIndexPair(4, 6);
    assignment[2] = TreeIndexPair(6, 10);

    SendList receiveList = computeHaloReceiveList(layout, haloFlags, assignment, peers);

    SendList reference(numRanks);
    reference[0].addRange(0, 1);
    reference[0].addRange(2, 4);
    reference[2].addRange(7, 9);

    EXPECT_EQ(receiveList, reference);
}
