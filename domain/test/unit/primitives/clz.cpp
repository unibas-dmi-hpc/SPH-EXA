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
 * @brief Count leading zero tests
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <vector>
#include "gtest/gtest.h"

#include "cstone/primitives/clz.hpp"

TEST(CLZ, C_clz_32)
{
    std::vector<unsigned> inputs{0xF0000000, 0x70000000, 1};
    std::vector<unsigned> references{0, 1, 31};

    std::vector<unsigned> probes;
    for (unsigned i : inputs)
        probes.push_back(detail::clz32(i));

    EXPECT_EQ(probes, references);
}

TEST(CLZ, C_clz_64)
{
    std::vector<uint64_t> inputs{1ul << 63u, (1ul << 62u) + 722302, 5, 4, 2, 1};
    std::vector<int> references{0, 1, 61, 61, 62, 63};

    std::vector<int> probes;
    for (auto i : inputs)
        probes.push_back(detail::clz64(i));

    EXPECT_EQ(probes, references);
}

TEST(CLZ, builtin_clz_32)
{
    std::vector<unsigned> inputs{0xF0000000, 0x70000000, 1, 0};
    std::vector<unsigned> references{0, 1, 31, 32};

    std::vector<unsigned> probes;
    for (unsigned i : inputs)
        probes.push_back(countLeadingZeros(i));

    EXPECT_EQ(probes, references);
}

TEST(CLZ, builtin_clz_64)
{
    std::vector<uint64_t> inputs{1ul << 63u, (1ul << 62u) + 23427, 5, 4, 2, 1, 0};
    std::vector<int> references{0, 1, 61, 61, 62, 63, 64};

    std::vector<int> probes;
    for (auto i : inputs)
        probes.push_back(countLeadingZeros(i));

    EXPECT_EQ(probes, references);
}
