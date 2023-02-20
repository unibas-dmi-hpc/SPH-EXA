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
 * @brief Test generic SFC functionality
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"
#include "cstone/sfc/sfc.hpp"

using namespace cstone;

TEST(SFC, commonNodePrefix)
{
    using T       = double;
    using KeyType = unsigned;

    Box<T> box(0, 1);

    {
        auto key = commonNodePrefix<HilbertKey<KeyType>>(Vec3<T>{0.7, 0.2, 0.2}, Vec3<T>{0.01, 0.01, 0.01}, box);
        EXPECT_EQ(key, 0000176134);
    }
    {
        auto key = commonNodePrefix<HilbertKey<KeyType>>(Vec3<T>{0.2393, 0.3272, 0.29372},
                                                         Vec3<T>{0.0012, 0.0011, 0.00098}, box);
        EXPECT_EQ(key, 0000104322);
    }
}

TEST(SFC, center)
{
    using T       = double;
    using KeyType = unsigned;

    Box<T> box(-1, 1);

    {
        // The exact center belongs to octant farthest from the origin
        T x           = 0.0;
        KeyType probe = sfc3D<HilbertKey<KeyType>>(x, x, x, box);
        KeyType ref   = sfc3D<HilbertKey<KeyType>>(1.0, 1.0, 1.0, box);
        EXPECT_EQ(octalDigit(probe, 1), octalDigit(ref, 1));
    }
    {
        // Center - epsilon should be in the octant closest to the origin
        T x           = -1e-40;
        KeyType probe = sfc3D<HilbertKey<KeyType>>(x, x, x, box);
        KeyType ref   = sfc3D<HilbertKey<KeyType>>(-1.0, -1.0, -1.0, box);
        EXPECT_EQ(octalDigit(probe, 1), octalDigit(ref, 1));
    }
}