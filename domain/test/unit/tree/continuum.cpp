/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich, University of Basel, University of Zurich
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
 * @brief Test continuum octree generation
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "cstone/tree/continuum.hpp"
#include "cstone/tree/cs_util.hpp"

using namespace cstone;

TEST(CsConcentration, concentrationCountConstant)
{
    using T       = double;
    using KeyType = uint32_t;

    Box<T> box(-1, 1);
    size_t N0 = 1000;

    auto constRho = [N0](T, T, T) { return T(N0) / 8.0; };

    size_t count = continuumCount(KeyType(0), nodeRange<KeyType>(0), box, constRho);

    EXPECT_EQ(count, N0);
}

TEST(CsConcentration, computeTreeOneOverR)
{
    using T       = double;
    using KeyType = uint64_t;

    unsigned bucketSize = 64;
    Box<T> box(-1, 1);
    T eps     = box.lx() / (1u << maxTreeLevel<KeyType>{});
    size_t N0 = 1000000;

    auto oneOverR = [N0, eps](T x, T y, T z)
    {
        T r = std::max(std::sqrt(norm2(Vec3<T>{x, y, z})), eps);
        if (r > 1.0) { return 0.0; }
        else { return T(N0) / (2 * M_PI * r); }
    };

    auto [tree, counts] = computeContinuumCsarray<KeyType>(oneOverR, box, bucketSize);
    size_t totalCount   = std::accumulate(counts.begin(), counts.end(), 0lu);

    for (auto c : counts)
    {
        EXPECT_LT(c, 1.5 * bucketSize);
    }

    EXPECT_NEAR(totalCount, N0, N0 * 0.03);
}
