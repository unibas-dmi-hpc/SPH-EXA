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
 * @brief Tests the SFC key exchange among peer ranks for halos
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <vector>

#include <gtest/gtest.h>

#include "cstone/domain/exchange_keys.hpp"
#include "cstone/domain/layout.hpp"

using namespace cstone;

/*! @brief exchange keys and test correctness of resulting send array indices
 *
 *  n = myRank * 4;
 *
 *  On each rank:
 *
 *  node index      0     1     2     3     4     5     6     7
 *  tree:        | n+0 | n+2 | n+4 | n+5 | n+6 | n+7 | n+8 | n+10 | n+12 |
 *  counts:         2     2     1     1     1     1     2     2
 *  haloFlag:             X                             X
 *  assignment   |-----------|-----------------------|-------------------|
 *                 myRank-1           myRank             myRank+1
 *  layout:         0     0     2     3     4     5     6     8      8
 *
 *  Rank i wants the halos with keys [n+2:n+4] from rank i-1 and [n+8:n+10] from rank i+1, n = 4 * i,
 *  and sends out those requests.
 *
 *  On rank i, we receive a request from rank i-1.
 *  Rank i-1 wants particles with keys [(n-4)+8:(n-4)+10] = [n+4:n+6].
 *  On rank i, these keys are in node indices 2-4, and the layout positions of nodes 2-4 is also 2-4.
 *  ==> rank i will send particles with index in [2:4] to rank i-1
 *
 *  On rank i we receive a request from rank i+1.
 *  Rank i+1 wants [(n+4)+2:(n+4)+4] = [n+6:n+8], which on rank i is nodes 4-6, also at layout position 4-6.
 *  ==> rank i will send particles with index in [4:6] to rank i+1
 */
template<class KeyType>
void exchangeKeys(int myRank, int numRanks)
{
    std::vector<unsigned> counts{2,2,1,1,1,1,2,2};
    std::vector<int> haloFlags{0,1,0,0,0,0,1,0};

    KeyType o = myRank * 4;
    std::vector<KeyType> treeLeaves{o, o+2, o+4, o+5, o+6, o+7, o+8, o+10, o+12};

    std::vector<LocalParticleIndex> layout = computeNodeLayout(counts, haloFlags, 2, 6);

    std::vector<TreeIndexPair> assignment(numRanks);
    std::vector<int>     peers;
    SendList             reference(numRanks);

    if (myRank > 0)
    {
        assignment[myRank - 1] = TreeIndexPair(0, 2);
        peers.push_back(myRank - 1);
        reference[myRank - 1].addRange(2, 4);
    }

    assignment[myRank] = TreeIndexPair(2, 6);

    if (myRank < numRanks - 1)
    {
        assignment[myRank + 1] = TreeIndexPair(6, 8);
        peers.push_back(myRank + 1);
        reference[myRank + 1].addRange(4, 6);
    }

    SendList probe = exchangeRequestKeys<KeyType>(treeLeaves, haloFlags, layout, assignment, peers);

    EXPECT_EQ(probe, reference);
}

TEST(ExchangeKeys, fixedTreeSizePerRank)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    exchangeKeys<unsigned>(rank, numRanks);
}
