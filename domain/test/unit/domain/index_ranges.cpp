/*
 * MIT License
 *
 * Copyright (c) 2022 CSCS, ETH Zurich
 *               2022 University of Basel
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
 * @brief Utility tests
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "cstone/domain/index_ranges.hpp"

using namespace cstone;

TEST(IndexRanges, empty)
{
    IndexRanges<int> ranges;
    EXPECT_EQ(ranges.nRanges(), 0);
    EXPECT_EQ(ranges.totalCount(), 0);
}

TEST(IndexRanges, addRange)
{
    IndexRanges<int> ranges;
    ranges.addRange(0, 5);

    EXPECT_EQ(ranges.nRanges(), 1);
    EXPECT_EQ(ranges.rangeStart(0), 0);
    EXPECT_EQ(ranges.rangeEnd(0), 5);
    EXPECT_EQ(ranges.count(0), 5);

    ranges.addRange(10, 19);

    EXPECT_EQ(ranges.nRanges(), 2);
    EXPECT_EQ(ranges.totalCount(), 14);

    EXPECT_EQ(ranges.rangeStart(1), 10);
    EXPECT_EQ(ranges.rangeEnd(1), 19);
    EXPECT_EQ(ranges.count(1), 9);
}
