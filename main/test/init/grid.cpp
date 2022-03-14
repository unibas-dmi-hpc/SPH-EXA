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
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <algorithm>

#include "gtest/gtest.h"

#include "init/grid.hpp"

using namespace sphexa;

TEST(Grids, intersect)
{
    using T = double;
    cstone::Box<T> box{0.12, 0.50, 0.26, 0.44, 0.55, 0.8};

    int multiplicity = 4;
    auto [l, u]      = gridIntersection(box, multiplicity);

    cstone::Vec3<int> refLower{0, 1, 2};
    cstone::Vec3<int> refUpper{2, 2, 4};

    EXPECT_EQ(l, refLower);
    EXPECT_EQ(u, refUpper);
}

TEST(Grids, scaleToGlobal)
{
    using T = double;
    cstone::Box<T> box{-1, 1};

    int multiplicity = 4;

    {
        cstone::Vec3<T> testX{0.0, 0.0, 0.0};
        auto            scaledX = scaleBlockToGlobal(testX, {0, 0, 0}, multiplicity, box);

        EXPECT_NEAR(scaledX[0], box.xmin(), 1e-10);
        EXPECT_NEAR(scaledX[1], box.ymin(), 1e-10);
        EXPECT_NEAR(scaledX[2], box.zmin(), 1e-10);
    }
    {
        cstone::Vec3<T> testX{0.0, 0.0, 0.0};
        auto            scaledX = scaleBlockToGlobal(testX, {2, 2, 2}, multiplicity, box);

        EXPECT_NEAR(scaledX[0], 0.0, 1e-10);
        EXPECT_NEAR(scaledX[1], 0.0, 1e-10);
        EXPECT_NEAR(scaledX[2], 0.0, 1e-10);
    }
    {
        cstone::Vec3<T> testX{1.0, 1.0, 1.0};
        auto            scaledX = scaleBlockToGlobal(testX, {3, 3, 3}, multiplicity, box);

        EXPECT_NEAR(scaledX[0], box.xmax(), 1e-10);
        EXPECT_NEAR(scaledX[1], box.ymax(), 1e-10);
        EXPECT_NEAR(scaledX[2], box.zmax(), 1e-10);
    }
    {
        cstone::Vec3<T> testX{0.1, 0.5, 0.6};
        auto            scaledX = scaleBlockToGlobal(testX, {0, 1, 2}, multiplicity, box);

        EXPECT_NEAR(scaledX[0], -0.95, 1e-10);
        EXPECT_NEAR(scaledX[1], -0.25, 1e-10);
        EXPECT_NEAR(scaledX[2], 0.3, 1e-10);
    }
}
