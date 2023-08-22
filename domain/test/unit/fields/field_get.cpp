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
 * @brief Tests for tuple gettres
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <array>

#include "gtest/gtest.h"

#include "cstone/fields/field_get.hpp"

using namespace cstone;

struct Testset
{
    inline static constexpr std::array fieldNames{"a", "b", "c"};
};

TEST(FieldGet, DatasetGet)
{
    auto tup = std::make_tuple(0, "alpha", 1.0);

    auto& e_a = get<"a", Testset>(tup);
    EXPECT_EQ(e_a, 0);

    e_a = 1;
    EXPECT_EQ(std::get<0>(tup), 1);
}

TEST(FieldGet, getPointers)
{
    std::vector<double> vd{1.0, 2.0, 3.0};
    std::vector<int> vi{1, 2, 3};

    int idx = 1;
    auto e1 = getPointers(std::tie(vd, vi), idx);

    *std::get<0>(e1) *= 2;
    *std::get<1>(e1) *= 3;

    EXPECT_EQ(vd[idx], 4.0);
    EXPECT_EQ(vi[idx], 6);
}
