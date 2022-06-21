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
#include <numeric>

#include "gtest/gtest.h"

#include "cstone/domain/domaindecomp.hpp"
#include "cstone/tree/octree.hpp"
#include "coord_samples/random.hpp"

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
void createSendList()
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
    createSendList<unsigned>();
    createSendList<uint64_t>();
}

template<class KeyType>
void extractRange()
{
    int bufferSize = 64;
    // the source array from which to extract the buffer
    std::vector<double> x(bufferSize);
    std::iota(begin(x), end(x), 0);

    SendManifest manifest;
    manifest.addRange(0, 8);
    manifest.addRange(40, 42);
    manifest.addRange(50, 50);

    std::vector<int> ordering(bufferSize);
    std::iota(begin(ordering), end(ordering), 0);

    // non-default ordering will make x appear sorted despite two elements being swapped
    std::swap(x[0], x[1]);
    std::swap(ordering[0], ordering[1]);

    std::vector<double> output(manifest.totalCount());
    extractRange(manifest, reinterpret_cast<char*>(x.data()), ordering.data(), reinterpret_cast<char*>(output.data()),
                 sizeof(double));

    // note sorted reference
    std::vector<double> ref{0, 1, 2, 3, 4, 5, 6, 7, 40, 41};
    EXPECT_EQ(output, ref);
}

TEST(DomainDecomposition, extractRange)
{
    extractRange<unsigned>();
    extractRange<uint64_t>();
}
