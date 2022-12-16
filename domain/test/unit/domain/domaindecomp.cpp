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
 * @brief Space filling curve octree assignment to ranks tests
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <algorithm>

#include "gtest/gtest.h"

#include "cstone/domain/domaindecomp.hpp"

using namespace cstone;

TEST(DomainDecomposition, singleRangeSfcSplit)
{
    {
        int nSplits = 2;
        std::vector<unsigned> counts{5, 5, 5, 5, 5, 6};

        auto splits = singleRangeSfcSplit(counts, nSplits);

        SpaceCurveAssignment ref(nSplits);
        ref.addRange(Rank(0), 0, 3, 15);
        ref.addRange(Rank(1), 3, 6, 16);
        EXPECT_EQ(ref, splits);
    }
    {
        int nSplits = 2;
        std::vector<unsigned> counts{5, 5, 5, 15, 1, 0};

        auto splits = singleRangeSfcSplit(counts, nSplits);

        SpaceCurveAssignment ref(nSplits);
        ref.addRange(Rank(0), 0, 3, 15);
        ref.addRange(Rank(1), 3, 6, 16);
        EXPECT_EQ(ref, splits);
    }
    {
        int nSplits = 2;
        std::vector<unsigned> counts{15, 0, 1, 5, 5, 5};

        auto splits = singleRangeSfcSplit(counts, nSplits);

        SpaceCurveAssignment ref(nSplits);
        ref.addRange(Rank(0), 0, 3, 16);
        ref.addRange(Rank(1), 3, 6, 15);
        EXPECT_EQ(ref, splits);
    }
    {
        int nSplits = 7;
        std::vector<unsigned> counts{4, 3, 4, 3, 4, 3, 4, 3, 4, 3};
        // should be grouped |4|7|3|7|4|7|3|

        auto splits = singleRangeSfcSplit(counts, nSplits);

        SpaceCurveAssignment ref(nSplits);
        ref.addRange(Rank(0), 0, 1, 4);
        ref.addRange(Rank(1), 1, 3, 7);
        ref.addRange(Rank(2), 3, 4, 3);
        ref.addRange(Rank(3), 4, 6, 7);
        ref.addRange(Rank(4), 6, 7, 4);
        ref.addRange(Rank(5), 7, 9, 7);
        ref.addRange(Rank(6), 9, 10, 3);
        EXPECT_EQ(ref, splits);
    }
}

//! @brief test that the SfcLookupKey can lookup the rank for a given code
TEST(DomainDecomposition, AssignmentFindRank)
{
    int nRanks = 4;
    SpaceCurveAssignment assignment(nRanks);
    assignment.addRange(Rank(0), 0, 1, 0);
    assignment.addRange(Rank(1), 1, 3, 0);
    assignment.addRange(Rank(2), 3, 4, 0);
    assignment.addRange(Rank(3), 4, 5, 0);

    EXPECT_EQ(0, assignment.findRank(0));
    EXPECT_EQ(1, assignment.findRank(1));
    EXPECT_EQ(1, assignment.findRank(2));
    EXPECT_EQ(2, assignment.findRank(3));
    EXPECT_EQ(3, assignment.findRank(4));
}

/*! @brief test SendList creation from a SFC assignment
 *
 * This test creates an array with SFC keys and an
 * SFC assignment with SFC keys ranges.
 * CreateSendList then translates the code ranges into indices
 * valid for the SFC key array.
 */
template<class KeyType>
static void sendListMinimal()
{
    std::vector<KeyType> tree{0, 2, 6, 8, 10};
    std::vector<KeyType> codes{0, 0, 1, 3, 4, 5, 6, 6, 9};

    int nRanks = 2;
    SpaceCurveAssignment assignment(nRanks);
    assignment.addRange(Rank(0), 0, 2, 0);
    assignment.addRange(Rank(1), 2, 4, 0);

    // note: codes input needs to be sorted
    auto sendList = createSendList<KeyType>(assignment, tree, codes);

    EXPECT_EQ(sendList[0].totalCount(), 6);
    EXPECT_EQ(sendList[1].totalCount(), 3);

    EXPECT_EQ(sendList[0].rangeStart(0), 0);
    EXPECT_EQ(sendList[0].rangeEnd(0), 6);
    EXPECT_EQ(sendList[1].rangeStart(0), 6);
    EXPECT_EQ(sendList[1].rangeEnd(0), 9);
}

TEST(DomainDecomposition, createSendList)
{
    sendListMinimal<unsigned>();
    sendListMinimal<uint64_t>();
}

TEST(DomainDecomposition, computeByteOffsets)
{
    util::array<size_t, 3> elementSizes{8, 4, 8};
    size_t sendCount = 1001;
    size_t alignment = 128;

    auto offsets = computeByteOffsets(sendCount, elementSizes, alignment);

    EXPECT_EQ(offsets[0], 0);
    EXPECT_EQ(offsets[1], round_up(elementSizes[0] * sendCount, alignment));
    EXPECT_EQ(offsets[2], offsets[1] + round_up(elementSizes[1] * sendCount, alignment));
    EXPECT_EQ(offsets[3], offsets[2] + round_up(elementSizes[2] * sendCount, alignment));

    EXPECT_EQ(offsets[3], 8064 + 4096 + 8064);
}

template<class KeyType>
static void initialSplits()
{
    auto ret = initialDomainSplits<KeyType>(3, 5);
    EXPECT_EQ(ret.front(), 0);
    EXPECT_EQ(ret.back(), nodeRange<KeyType>(0));
}

TEST(DomainDecomposition, initialDomainSplit)
{
    initialSplits<unsigned>();
    initialSplits<uint64_t>();
}