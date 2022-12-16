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
 * @brief Global octree build test
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <vector>
#include <mpi.h>

#include <gtest/gtest.h>

#include "cstone/tree/update_mpi.hpp"
#include "cstone/tree/cs_util.hpp"

using namespace cstone;

template<class I>
void buildTree(int rank)
{
    constexpr unsigned level = 2;
    std::vector<I> allCodes  = makeNLevelGrid<I>(level);
    std::vector<I> codes{begin(allCodes) + rank * allCodes.size() / 2,
                         begin(allCodes) + (rank + 1) * allCodes.size() / 2};

    int bucketSize = 8;

    std::vector<I> tree = makeRootNodeTree<I>();
    std::vector<unsigned> counts{unsigned(codes.size())};
    while (!updateOctreeGlobal(codes.data(), codes.data() + codes.size(), bucketSize, tree, counts))
        ;

    std::vector<I> refTree = OctreeMaker<I>{}.divide().makeTree();

    std::vector<unsigned> refCounts(8, 8);

    EXPECT_EQ(counts, refCounts);
    EXPECT_EQ(tree, refTree);
}

TEST(GlobalTree, basicRegularTree32)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    constexpr int thisExampleRanks = 2;

    if (nRanks != thisExampleRanks) throw std::runtime_error("this test needs 2 ranks\n");

    buildTree<unsigned>(rank);
    buildTree<uint64_t>(rank);
}
