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
 * @brief Test box functionality
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"
#include "cstone/sfc/box.hpp"

using namespace cstone;

TEST(SfcBox, pbcAdjust)
{
    EXPECT_EQ(pbcAdjust<1024>(-1024), 0);
    EXPECT_EQ(pbcAdjust<1024>(-1), 1023);
    EXPECT_EQ(pbcAdjust<1024>(0), 0);
    EXPECT_EQ(pbcAdjust<1024>(1), 1);
    EXPECT_EQ(pbcAdjust<1024>(1023), 1023);
    EXPECT_EQ(pbcAdjust<1024>(1024), 0);
    EXPECT_EQ(pbcAdjust<1024>(1025), 1);
    EXPECT_EQ(pbcAdjust<1024>(2047), 1023);
}

TEST(SfcBox, pbcDistance)
{
    EXPECT_EQ(pbcDistance<1024>(-1024), 0);
    EXPECT_EQ(pbcDistance<1024>(-513), 511);
    EXPECT_EQ(pbcDistance<1024>(-512), 512);
    EXPECT_EQ(pbcDistance<1024>(-1), -1);
    EXPECT_EQ(pbcDistance<1024>(0), 0);
    EXPECT_EQ(pbcDistance<1024>(1), 1);
    EXPECT_EQ(pbcDistance<1024>(512), 512);
    EXPECT_EQ(pbcDistance<1024>(513), -511);
    EXPECT_EQ(pbcDistance<1024>(1024), 0);
}

TEST(SfcBox, applyPbc)
{
    using T = double;

    Box<T> box(0, 1, BoundaryType::periodic);
    Vec3<T> X{0.9, 0.9, 0.9};
    auto Xpbc = cstone::applyPbc(X, box);

    EXPECT_NEAR(Xpbc[0], -0.1, 1e-10);
    EXPECT_NEAR(Xpbc[1], -0.1, 1e-10);
    EXPECT_NEAR(Xpbc[2], -0.1, 1e-10);
}

TEST(SfcBox, putInBox)
{
    using T = double;
    {
        Box<T> box(0, 1, BoundaryType::periodic);
        Vec3<T> X{0.9, 0.9, 0.9};
        auto Xpbc = cstone::putInBox(X, box);

        EXPECT_NEAR(Xpbc[0], 0.9, 1e-10);
        EXPECT_NEAR(Xpbc[1], 0.9, 1e-10);
        EXPECT_NEAR(Xpbc[2], 0.9, 1e-10);
    }
    {
        Box<T> box(0, 1, BoundaryType::periodic);
        Vec3<T> X{1.1, 1.1, 1.1};
        auto Xpbc = cstone::putInBox(X, box);

        EXPECT_NEAR(Xpbc[0], 0.1, 1e-10);
        EXPECT_NEAR(Xpbc[1], 0.1, 1e-10);
        EXPECT_NEAR(Xpbc[2], 0.1, 1e-10);
    }
    {
        Box<T> box(-1, 1, BoundaryType::periodic);
        Vec3<T> X{-0.9, -0.9, -0.9};
        auto Xpbc = cstone::putInBox(X, box);

        EXPECT_NEAR(Xpbc[0], -0.9, 1e-10);
        EXPECT_NEAR(Xpbc[1], -0.9, 1e-10);
        EXPECT_NEAR(Xpbc[2], -0.9, 1e-10);
    }
}

TEST(SfcBox, createIBox)
{
    {
        using T                = double;
        using KeyType          = uint32_t;
        constexpr int maxCoord = 1u << maxTreeLevel<KeyType>{};

        Box<T> box(0, 1);

        T r = T(1.0) / maxCoord;
        T c = 1.0 - 0.5 * r;
        T s = 0.5 * r;
        Vec3<T> aCenter{c, c, c};
        Vec3<T> aSize{s, s, s};

        IBox probe = createIBox<KeyType>(aCenter, aSize, box);
        IBox ref{maxCoord - 1, maxCoord};
        EXPECT_EQ(ref, probe);
    }
    {
        using T       = double;
        using KeyType = uint64_t;

        Box<T> box(-1, 1, -2, 2, -3, 3);
        Vec3<T> aCenter{0.1, 0.2, 0.3};
        Vec3<T> aSize{0.01, 0.02, 0.03};

        IBox probe = createIBox<KeyType>(aCenter, aSize, box);
        IBox ref{1142947, 1163920};
        EXPECT_EQ(ref, probe);
    }
}