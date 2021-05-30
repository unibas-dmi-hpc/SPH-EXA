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
 * @brief Focus exchange test
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <gtest/gtest.h>

#include "cstone/tree/focus_exchange.hpp"
#include "cstone/tree/octree_util.hpp"

using namespace cstone;

template<class I>
void exchangeFocus(int myRank)
{
    std::vector<I>        treeLeaves = makeUniformNLevelTree<I>(64, 1);
    std::vector<unsigned> counts(nNodes(treeLeaves), myRank + 1);

    std::vector<int>                 peers;
    std::vector<pair<TreeNodeIndex>> peerFocusIndices;

    if (myRank == 0)
    {
        peers.push_back(1);
        peerFocusIndices.emplace_back(32, 64);
    }
    else
    {
        peers.push_back(0);
        peerFocusIndices.emplace_back(0, 32);
    }

    std::vector<I>        tmpLeaves(32 + 1);
    std::vector<unsigned> tmpCounts(32);

    exchangeFocus(peers, peerFocusIndices, treeLeaves, counts, tmpLeaves, tmpCounts);

    std::vector<unsigned> reference(nNodes(treeLeaves), myRank + 1);
    if (myRank == 0)
    {
        std::fill(begin(reference) + 32, end(reference), 2);
    }
    else
    {
        std::fill(begin(reference), begin(reference) + 32, 1);
    }

    EXPECT_EQ(counts, reference);
}

TEST(FocusExchange, simpleTest)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    constexpr int thisExampleRanks = 2;

    if (nRanks != thisExampleRanks)
        throw std::runtime_error("this test needs 2 ranks\n");

    exchangeFocus<unsigned>(rank);
}