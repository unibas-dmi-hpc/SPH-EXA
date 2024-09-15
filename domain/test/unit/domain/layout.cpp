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
#include "cstone/tree/cs_util.hpp"

using namespace cstone;

TEST(DomainDecomposition, invertRanges)
{
    {
        TreeNodeIndex first = 0;
        TreeNodeIndex last  = 10;

        std::vector<TreeIndexPair> ranges{{0, 0}, {0, 0}, {1, 2}, {2, 3}, {0, 0}, {5, 8}, {0, 0}};

        std::vector<TreeIndexPair> ref{{0, 1}, {3, 5}, {8, 10}};

        auto probe = invertRanges(first, ranges, last);
        EXPECT_EQ(probe, ref);
    }
    {
        TreeNodeIndex first = 0;
        TreeNodeIndex last  = 10;

        std::vector<TreeIndexPair> ranges{{0, 2}, {2, 3}, {5, 8}};

        std::vector<TreeIndexPair> ref{{3, 5}, {8, 10}};

        auto probe = invertRanges(first, ranges, last);
        EXPECT_EQ(probe, ref);
    }
    {
        TreeNodeIndex first = 0;
        TreeNodeIndex last  = 10;

        std::vector<TreeIndexPair> ranges{{0, 2}, {2, 3}, {5, 10}};

        std::vector<TreeIndexPair> ref{{3, 5}};

        auto probe = invertRanges(first, ranges, last);
        EXPECT_EQ(probe, ref);
    }
}

//! @brief tests extraction of SFC keys for all nodes marked as halos within an index range
TEST(Layout, extractMarkedElements)
{
    std::vector<unsigned> leaves{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<int> haloFlags{0, 0, 0, 1, 1, 1, 0, 1, 0, 1};

    {
        std::vector<unsigned> reqKeys = extractMarkedElements<unsigned>(leaves, haloFlags, 0, 0);
        std::vector<unsigned> reference{};
        EXPECT_EQ(reqKeys, reference);
    }
    {
        std::vector<unsigned> reqKeys = extractMarkedElements<unsigned>(leaves, haloFlags, 0, 3);
        std::vector<unsigned> reference{};
        EXPECT_EQ(reqKeys, reference);
    }
    {
        std::vector<unsigned> reqKeys = extractMarkedElements<unsigned>(leaves, haloFlags, 0, 4);
        std::vector<unsigned> reference{3, 4};
        EXPECT_EQ(reqKeys, reference);
    }
    {
        std::vector<unsigned> reqKeys = extractMarkedElements<unsigned>(leaves, haloFlags, 0, 5);
        std::vector<unsigned> reference{3, 5};
        EXPECT_EQ(reqKeys, reference);
    }
    {
        std::vector<unsigned> reqKeys = extractMarkedElements<unsigned>(leaves, haloFlags, 0, 7);
        std::vector<unsigned> reference{3, 6};
        EXPECT_EQ(reqKeys, reference);
    }
    {
        std::vector<unsigned> reqKeys = extractMarkedElements<unsigned>(leaves, haloFlags, 0, 10);
        std::vector<unsigned> reference{3, 6, 7, 8, 9, 10};
        EXPECT_EQ(reqKeys, reference);
    }
    {
        std::vector<unsigned> reqKeys = extractMarkedElements<unsigned>(leaves, haloFlags, 9, 10);
        std::vector<unsigned> reference{9, 10};
        EXPECT_EQ(reqKeys, reference);
    }
}

TEST(Layout, computeHaloReceiveList)
{
    std::vector<LocalIndex> layout{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
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

TEST(Layout, gatherArrays)
{
    std::vector<LocalIndex> ordering{1, 0, 2, 3};
    std::vector<float> a{0., 1., 2., 3., 4.};
    std::vector<unsigned char> b{0, 1, 2, 3, 4};

    std::vector<float> scratch(a.size());

    LocalIndex inOffset = 1;
    LocalIndex outOffset = 1;
    gatherArrays(gatherCpu, ordering.data(), ordering.size(), inOffset, outOffset, std::tie(a, b), std::tie(scratch));

    static_assert(not SmallerElementSize<0, std::vector<int>, std::tuple<std::vector<char>, std::vector<int>>>{});
    static_assert(SmallerElementSize<1, std::vector<int>, std::tuple<std::vector<char>, std::vector<int>>>{});

    std::vector<float> refA{0, 2., 1., 3., 4.};
    std::vector<unsigned char> refB{0, 2, 1, 3, 4};

    EXPECT_TRUE(std::equal(&refA[outOffset], &refA[a.size()], &a[outOffset]));
    EXPECT_TRUE(std::equal(&refB[outOffset], &refB[b.size()], &b[outOffset]));
}

TEST(Layout, enumerateRanges)
{
    std::vector<IndexPair<TreeNodeIndex>> ranges{{10, 13}, {30, 32}};
    auto probe = enumerateRanges(ranges);
    std::vector<TreeNodeIndex> ref{10, 11, 12, 30, 31};
    EXPECT_EQ(probe, ref);
}