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
 * @brief Test cpu gather functionality
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <vector>

#include "gtest/gtest.h"

#include "cstone/primitives/gather.hpp"

using namespace cstone;

TEST(GatherCpu, sortInvert)
{
    std::vector<int> keys{2, 1, 5, 4};

    // the ordering that sorts keys is {1,0,3,2}
    std::vector<int> values(keys.size());
    std::iota(begin(values), end(values), 0);

    sort_by_key(begin(keys), end(keys), begin(values));

    std::vector<int> reference{1, 0, 3, 2};
    EXPECT_EQ(values, reference);
}

template<class ValueType, class KeyType, class IndexType>
void CpuGatherTest()
{
    std::vector<KeyType> codes{0, 50, 10, 60, 20, 70, 30, 80, 40, 90};

    std::vector<unsigned> scratch;
    SfcSorter<IndexType, std::vector<unsigned>> sorter(scratch);
    sorter.setMapFromCodes(codes.data(), codes.data() + codes.size());

    {
        std::vector<KeyType> refCodes{0, 10, 20, 30, 40, 50, 60, 70, 80, 90};
        EXPECT_EQ(codes, refCodes);
    }

    std::vector<ValueType> values{-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    std::vector<ValueType> probe = values;
    gatherCpu(sorter.getMap(), codes.size(), values.data() + 2, probe.data() + 2);
    std::vector<ValueType> reference{-2, -1, 0, 2, 4, 6, 8, 1, 3, 5, 7, 9, 10, 11};

    EXPECT_EQ(probe, reference);
}

TEST(GatherCpu, CpuGather)
{
    CpuGatherTest<float, unsigned, unsigned>();
    CpuGatherTest<float, uint64_t, unsigned>();
    CpuGatherTest<double, unsigned, unsigned>();
    CpuGatherTest<double, uint64_t, unsigned>();
}