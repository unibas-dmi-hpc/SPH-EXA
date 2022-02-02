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
#include "cstone/tree/octree_util.hpp"

using namespace cstone;

/*! @brief simple particle count exchange test with 2 ranks
 *
 * Each ranks has a regular level-2 grid with 64 elements as tree.
 * Rank 0 queries Rank 1 for particle counts in the x,y,z = [0.5-1, 0-1, 0-1] half
 * and vice versa.
 */
template<class KeyType>
void exchangeFocus(int myRank, int numRanks)
{
    std::vector<KeyType>  treeLeaves = makeUniformNLevelTree<KeyType>(64, 1);
    std::vector<unsigned> counts(nNodes(treeLeaves), 0);

    std::vector<int> peers;
    std::vector<IndexPair<TreeNodeIndex>> peerFocusIndices(2);

    if (myRank == 0)
    {
        peers.push_back(1);
        peerFocusIndices[1] = TreeIndexPair(32, 64);
        std::fill(begin(counts), begin(counts) + 32, 1);
    }
    else
    {
        peers.push_back(0);
        peerFocusIndices[0] = TreeIndexPair(0, 32);
        std::fill(begin(counts) + 32, end(counts), 1);
    }

    exchangePeerCounts<KeyType>(peers, peerFocusIndices, treeLeaves, counts);

    std::vector<unsigned> reference(nNodes(treeLeaves), 1);

    EXPECT_EQ(counts, reference);
}

TEST(PeerExchange, simpleTest)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    constexpr int thisExampleRanks = 2;

    if (numRanks != thisExampleRanks) throw std::runtime_error("this test needs 2 ranks\n");

    exchangeFocus<unsigned>(rank, numRanks);
    exchangeFocus<uint64_t>(rank, numRanks);
}

/*! @brief irregular tree particle count exchange with 2 ranks
 *
 * In this test, each rank has a regular level-3 grid in its assigned half
 * of the cube with 512/2 = 256 elements. Outside the assigned area,
 * the tree structure is irregular.
 *
 * Rank 0 still queries Rank 1 for particle counts in the x,y,z = [0.5-1, 0-1, 0-1] half
 * and vice versa, but the tree structure that rank 0 has and sends to Rank 1 differs
 * from the regular grid that rank 1 has in this half.
 */
template<class KeyType>
void exchangeFocusIrregular(int myRank, int numRanks)
{
    std::vector<KeyType> treeLeaves;
    std::vector<int> peers;
    std::vector<IndexPair<TreeNodeIndex>> peerFocusIndices(numRanks);

    OctreeMaker<KeyType> octreeMaker;
    octreeMaker.divide();
    if (myRank == 0)
    {
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
        treeLeaves = octreeMaker.makeTree();

        peers.push_back(1);
        TreeNodeIndex peerStartIdx =
            std::lower_bound(begin(treeLeaves), end(treeLeaves), codeFromIndices<KeyType>({4})) - begin(treeLeaves);
        peerFocusIndices[1] = TreeIndexPair(peerStartIdx, nNodes(treeLeaves));
    }
    else
    {
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
        treeLeaves = octreeMaker.makeTree();

        peers.push_back(0);
        TreeNodeIndex peerEndIdx =
            std::lower_bound(begin(treeLeaves), end(treeLeaves), codeFromIndices<KeyType>({4})) - begin(treeLeaves);
        peerFocusIndices[0] = TreeIndexPair(0, peerEndIdx);
    }

    std::vector<unsigned> counts(nNodes(treeLeaves), 1);

    exchangePeerCounts<KeyType>(peers, peerFocusIndices, treeLeaves, counts);

    std::vector<unsigned> reference(nNodes(treeLeaves), 1);
    TreeNodeIndex peerStartIdx, peerEndIdx;
    if (myRank == 0)
    {
        peerStartIdx =
            std::lower_bound(begin(treeLeaves), end(treeLeaves), codeFromIndices<KeyType>({4})) - begin(treeLeaves);
        peerEndIdx = nNodes(treeLeaves);
    }
    else
    {
        peerStartIdx = 0;
        peerEndIdx = std::lower_bound(begin(treeLeaves), end(treeLeaves), codeFromIndices<KeyType>({4})) - begin(treeLeaves);
    }

    for (int i = peerStartIdx; i < peerEndIdx; ++i)
    {
        int level = treeLevel(treeLeaves[i + 1] - treeLeaves[i]);
        // the particle count per node outside the focus is 8^(3 - level-of-node)
        unsigned numParticles = 1u << (3 * (3 - level));
        reference[i] = numParticles;
    }

    EXPECT_EQ(counts, reference);
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
