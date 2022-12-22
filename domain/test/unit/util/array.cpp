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

/*! @file @brief This implements basic util::array tests
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"
#include "cstone/util/array.hpp"

namespace util
{

TEST(Array, construct)
{
    util::array<int, 3> a{0, 1, 2};

    EXPECT_EQ(a[0], 0);
    EXPECT_EQ(a[1], 1);
    EXPECT_EQ(a[2], 2);
}

TEST(Array, plusEqual)
{
    util::array<int, 3> a{0, 1, 2};
    util::array<int, 3> b{1, 2, 3};

    a += b;

    EXPECT_EQ(a[0], 1);
    EXPECT_EQ(a[1], 3);
    EXPECT_EQ(a[2], 5);
}

TEST(Array, minusEqual)
{
    util::array<int, 3> a{0, 1, 2};
    util::array<int, 3> b{1, 0, 3};

    a -= b;

    EXPECT_EQ(a[0], -1);
    EXPECT_EQ(a[1], 1);
    EXPECT_EQ(a[2], -1);
}

TEST(Array, equal)
{
    util::array<int, 3> a{0, 1, 2};
    util::array<int, 3> b{1, 2, 3};

    util::array<int, 3> c{1, 2, 3};

    EXPECT_FALSE(a == b);
    EXPECT_TRUE(b == c);
}

TEST(Array, unequal)
{
    util::array<int, 3> a{0, 1, 2};
    util::array<int, 3> b{1, 2, 3};
    util::array<int, 3> c{1, 2, 3};

    EXPECT_TRUE(a != b);
    EXPECT_FALSE(b != c);
}

TEST(Array, smaller)
{
    {
        util::array<int, 3> a{0, 0, 0};
        util::array<int, 3> b{0, 0, 0};
        EXPECT_FALSE(a < b);
    }
    {
        util::array<int, 3> a{0, 0, 0};
        util::array<int, 3> b{1, 0, 0};
        EXPECT_TRUE(a < b);
        EXPECT_FALSE(b < a);
    }
    {
        util::array<int, 3> a{0, 0, 0};
        util::array<int, 3> b{0, 1, 0};
        EXPECT_TRUE(a < b);
        EXPECT_FALSE(b < a);
    }
    {
        util::array<int, 3> a{0, 0, 0};
        util::array<int, 3> b{0, 0, 1};
        EXPECT_TRUE(a < b);
        EXPECT_FALSE(b < a);
    }
    {
        util::array<int, 3> a{2, 4, 0};
        util::array<int, 3> b{3, 0, 0};
        EXPECT_TRUE(a < b);
        EXPECT_FALSE(b < a);
    }
}

TEST(Array, scalarMultiply)
{
    util::array<int, 3> a{2, 4, 6};
    a *= 2;
    EXPECT_EQ(a[0], 4);
    EXPECT_EQ(a[1], 8);
    EXPECT_EQ(a[2], 12);
}

TEST(Array, scalarDivide)
{
    util::array<int, 3> a{2, 4, 6};
    a /= 2;
    EXPECT_EQ(a[0], 1);
    EXPECT_EQ(a[1], 2);
    EXPECT_EQ(a[2], 3);
}

TEST(Array, binaryAdd)
{
    util::array<int, 3> a{2, 4, 0};
    util::array<int, 3> b{3, 0, 1};

    util::array<int, 3> s{5, 4, 1};
    EXPECT_EQ(s, a + b);
}

TEST(Array, binarySub)
{
    util::array<int, 3> a{2, 4, 0};
    util::array<int, 3> b{3, 0, 1};

    util::array<int, 3> d{-1, 4, -1};
    EXPECT_EQ(d, a - b);
}

TEST(Array, freeScalarMultiply)
{
    util::array<int, 3> a{2, 4, 0};

    util::array<int, 3> p{4, 8, 0};
    EXPECT_EQ(p, a * 2);
}

TEST(Array, dot)
{
    util::array<int, 3> a{2, 4, 2};
    util::array<int, 3> b{4, 8, 1};
    EXPECT_EQ(dot(a, b), 42);
}

TEST(Array, negate)
{
    util::array<int, 3> a{2, 4, 2};
    util::array<int, 3> b = -a;

    util::array<int, 3> ref{-2, -4, -2};
    EXPECT_EQ(b, ref);
}

TEST(Array, assignValue)
{
    util::array<int, 3> a;

    a = 1;
    util::array<int, 3> ref{1, 1, 1};
    EXPECT_EQ(a, ref);
}

TEST(Array, structuredBinding)
{
    util::array<int, 2> a{1, 2};
    auto& [x, y] = a;
    auto [u, v]  = a;

    x = 3;
    EXPECT_EQ(get<0>(a), 3);
    EXPECT_EQ(u, 1);
}

TEST(Array, reduction)
{
    util::array<size_t, 3ul> a{1ul, 2ul, 3ul};

    size_t numElements = 10000;
    std::vector v(numElements, a);

    util::array<size_t, 3> sum{0ul, 0ul, 0ul};

#pragma omp declare reduction(+ : array <size_t, 3> : omp_out += omp_in) initializer(omp_priv(omp_orig))

#pragma omp parallel for reduction(+ : sum)
    for (size_t i = 0; i < numElements; ++i)
    {
        sum += v[i];
    }

    EXPECT_EQ(sum, numElements * a);
}

} // namespace util
