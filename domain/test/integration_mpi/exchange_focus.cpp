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

#include "cstone/focus/exchange_focus.hpp"
#include "cstone/tree/cs_util.hpp"

using namespace cstone;

/*! @brief irregular treelet exchange with 2 ranks
 *
 * In this test, each rank has a regular level-3 grid in its assigned half
 * of the cube with 512/2 = 256 elements. Outside the assigned area,
 * the tree structure is irregular.
 */
template<class KeyType>
void exchangeFocusIrregular(int myRank, int numRanks)
{
    std::vector<KeyType> treeLeavesRef[numRanks];
    std::vector<int> peers;
    std::vector<IndexPair<TreeNodeIndex>> peerFocusIndices(numRanks);

    // create reference trees
    {
        OctreeMaker<KeyType> octreeMaker;
        octreeMaker.divide();
        // regular level-3 grid in the half cube with x = 0...0.5
        for (int i = 0; i < 4; ++i)
        {
            octreeMaker.divide({i}, 1);
            for (int j = 0; j < 8; ++j)
            {
                octreeMaker.divide({i, j}, 2);
            }
        }
        // finer resolution at one location outside the regular grid
        octreeMaker.divide(7).divide(7, 0);
        treeLeavesRef[0] = octreeMaker.makeTree();
    }
    {
        OctreeMaker<KeyType> octreeMaker;
        octreeMaker.divide();
        // regular level-3 grid in the half cube with x = 0.5...1
        for (int i = 4; i < 8; ++i)
        {
            octreeMaker.divide({i}, 1);
            for (int j = 0; j < 8; ++j)
            {
                octreeMaker.divide({i, j}, 2);
            }
        }
        // finer resolution at one location outside the regular grid
        octreeMaker.divide(1).divide(1, 6);
        treeLeavesRef[1] = octreeMaker.makeTree();
    }

    std::vector<KeyType> treeLeaves = treeLeavesRef[myRank];

    if (myRank == 0)
    {
        peers.push_back(1);
        TreeNodeIndex peerStartIdx =
            std::lower_bound(begin(treeLeaves), end(treeLeaves), codeFromIndices<KeyType>({4})) - begin(treeLeaves);
        peerFocusIndices[1] = TreeIndexPair(peerStartIdx, nNodes(treeLeaves));
    }
    else
    {
        peers.push_back(0);
        TreeNodeIndex peerEndIdx =
            std::lower_bound(begin(treeLeaves), end(treeLeaves), codeFromIndices<KeyType>({4})) - begin(treeLeaves);
        peerFocusIndices[0] = TreeIndexPair(0, peerEndIdx);
    }

    std::vector<std::vector<KeyType>> treelets(numRanks);
    std::vector<MPI_Request> requests;
    exchangeTreelets<KeyType>(peers, peerFocusIndices, treeLeaves, treelets, requests);
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUS_IGNORE);

    if (myRank == 0) { EXPECT_TRUE(std::equal(begin(treelets[1]), end(treelets[1]), treeLeavesRef[1].begin())); }
    else
    {
        TreeNodeIndex peerStartIdx =
            std::lower_bound(begin(treeLeavesRef[0]), end(treeLeavesRef[0]), codeFromIndices<KeyType>({4})) -
            begin(treeLeavesRef[0]);
        EXPECT_TRUE(std::equal(begin(treelets[0]), end(treelets[0]), treeLeavesRef[0].begin() + peerStartIdx));
    }
}

TEST(PeerExchange, irregularTree)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    constexpr int thisExampleRanks = 2;

    if (numRanks != thisExampleRanks) throw std::runtime_error("this test needs 2 ranks\n");

    exchangeFocusIrregular<unsigned>(rank, numRanks);
    exchangeFocusIrregular<uint64_t>(rank, numRanks);
}

TEST(PeerExchange, arrayWrap)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    using Vec = util::array<uint64_t, 4>;

    if (rank == 0)
    {
        std::vector<Vec> buffer{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}};

        std::vector<MPI_Request> requests;
        mpiSendAsync(buffer.data(), buffer.size(), 1, 0, requests);
        MPI_Waitall(int(requests.size()), requests.data(), MPI_STATUS_IGNORE);
    }
    if (rank == 1)
    {
        std::vector<Vec> buffer(3);
        mpiRecvSync(buffer.data(), buffer.size(), 0, 0, MPI_STATUS_IGNORE);

        std::vector<Vec> reference{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}};
        EXPECT_EQ(buffer, reference);
    }
}
