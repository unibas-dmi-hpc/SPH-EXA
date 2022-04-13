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
 * @brief Unit tests for grid-based initialization functionality
 *
 * @author Jose A. Escartin <ja.escartin@gmail.com>
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "cstone/sfc/common.hpp"

#include "init/isobaric_cube_init.hpp"

using namespace sphexa;

TEST(IsobaricCube, pyramidStretch)
{
    using T = double;

    T h = 0.25;
    T s = 0.40548013;
    T r = 0.5;

    {
        cstone::Vec3<T> X{s, 0, 0};
        auto            Xp = X * cappedPyramidStretch(X, h, s, r);
        EXPECT_NEAR(Xp[0], h, 1e-6);
        EXPECT_NEAR(Xp[1], 0, 1e-6);
        EXPECT_NEAR(Xp[2], 0, 1e-6);
    }
    { // point on the surface of the stretch boundary stays
        cstone::Vec3<T> X{s, 0.1, 0.2};
        auto            Xp = X * cappedPyramidStretch(X, h, s, r);
        EXPECT_NEAR(Xp[0], h, 1e-6);
        EXPECT_LT(Xp[1], X[1]);
        EXPECT_LT(Xp[2], X[2]);
    }
    { // point on surface of stretch-boundary gets mapped to the surface of the internal cube
        cstone::Vec3<T> X{s, s, s};
        auto            Xp = X * cappedPyramidStretch(X, h, s, r);
        EXPECT_NEAR(Xp[0], h, 1e-6);
        EXPECT_NEAR(Xp[1], h, 1e-6);
        EXPECT_NEAR(Xp[2], h, 1e-6);
    }
    { // point on the surface of the outer cube stays on the outer surface
        cstone::Vec3<T> X{r, r, r};
        auto            Xp = X * cappedPyramidStretch(X, h, s, r);
        EXPECT_NEAR(Xp[0], r, 1e-6);
        EXPECT_NEAR(Xp[1], r, 1e-6);
        EXPECT_NEAR(Xp[2], r, 1e-6);
    }
    { // point on the surface of the outer cube stays on the outer surface
        cstone::Vec3<T> X{r, 0.1, 0.2};
        auto            Xp = X * cappedPyramidStretch(X, h, s, r);
        EXPECT_NEAR(Xp[0], r, 1e-6);
        EXPECT_NEAR(Xp[1], 0.1, 1e-6);
        EXPECT_NEAR(Xp[2], 0.2, 1e-6);
    }
}

TEST(IsobaricCube, stretchFactor)
{
    using T = double;

    T h        = 0.25;
    T r        = 0.5;
    T rhoRatio = 8;

    T s = computeStretchFactor(h, r, rhoRatio);

    T sRef = 0.40548013;
    EXPECT_NEAR(s, sRef, 1e-6);
}