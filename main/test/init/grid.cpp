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

#include "cstone/sfc/common.hpp"

#include "init/grid.hpp"

using namespace sphexa;

TEST(Grids, intersect)
{
    using T = double;
    cstone::FBox<T> box{0.12, 0.50, 0.26, 0.44, 0.55, 0.8};

    std::tuple<int, int, int> multiplicity = {4, 4, 4};
    auto [l, u]                            = gridIntersection(box, multiplicity);

    cstone::Vec3<int> refLower{0, 1, 2};
    cstone::Vec3<int> refUpper{2, 2, 4};

    EXPECT_EQ(l, refLower);
    EXPECT_EQ(u, refUpper);
}

TEST(Grids, scaleToGlobal)
{
    using T = double;
    cstone::Box<T> box{-1, 1};

    std::tuple<int, int, int> multiplicity = {4, 4, 4};

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

TEST(Grids, assembleRectangle)
{
    using T       = double;
    using KeyType = unsigned;
    cstone::Box<T> box{-1, 1};

    std::tuple<int, int, int> multiplicity = {20, 20, 20};

    std::vector<T> xb{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
    std::vector<T> yb{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
    std::vector<T> zb{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};

    // 3 SFC segments: 0-k1 k1-k2 k2-nodeRange(0)
    KeyType k1 = 01234567012;
    KeyType k2 = 05123456701;

    std::vector<T> x1, y1, z1;
    assembleRectangle<T>(KeyType(0), k1, box, multiplicity, xb, yb, zb, x1, y1, z1);

    std::vector<T> x2, y2, z2;
    assembleRectangle<T>(k1, k2, box, multiplicity, xb, yb, zb, x2, y2, z2);

    std::vector<T> x3, y3, z3;
    assembleRectangle<T>(k2, cstone::nodeRange<KeyType>(0), box, multiplicity, xb, yb, zb, x3, y3, z3);

    // total number of particles in the 3 segments together should be initBlock.size() * multiplicity^3
    size_t totalSize = x1.size() + x2.size() + x3.size();
    EXPECT_EQ(totalSize, std::get<0>(multiplicity) * std::get<1>(multiplicity) * std::get<2>(multiplicity) * xb.size());

    std::vector<KeyType> keys(totalSize);
    auto                 ksfc = cstone::sfcKindPointer(keys.data());

    cstone::computeSfcKeys(x1.data(), y1.data(), z1.data(), ksfc, x1.size(), box);
    cstone::computeSfcKeys(x2.data(), y2.data(), z2.data(), ksfc + x1.size(), x2.size(), box);
    cstone::computeSfcKeys(x3.data(), y3.data(), z3.data(), ksfc + x1.size() + x2.size(), x3.size(), box);

    // explicit construction
    std::vector<T> X, Y, Z;
    for (int i = 0; i < std::get<0>(multiplicity); i++)
    {
        for (int j = 0; j < std::get<1>(multiplicity); ++j)
        {
            for (int k = 0; k < std::get<2>(multiplicity); ++k)
            {
                auto selectBox = cstone::FBox<T>(
                    box.lx() / std::get<0>(multiplicity) * i - 1, box.lx() / std::get<0>(multiplicity) * (i + 1) - 1,
                    box.ly() / std::get<1>(multiplicity) * j - 1, box.ly() / std::get<1>(multiplicity) * (j + 1) - 1,
                    box.lz() / std::get<2>(multiplicity) * k - 1, box.lz() / std::get<2>(multiplicity) * (k + 1) - 1);
                extractBlock(selectBox, box, {i, j, k}, multiplicity, (gsl::span<const T>)xb, (gsl::span<const T>)yb,
                             (gsl::span<const T>)zb, X, Y, Z);
            }
        }
    }

    std::vector<KeyType> keys2(X.size());
    auto                 ksfc2 = cstone::sfcKindPointer(keys2.data());
    cstone::computeSfcKeys(X.data(), Y.data(), Z.data(), ksfc2, X.size(), box);


    for (size_t i = 0; i < x1.size(); ++i)
    {
        EXPECT_TRUE(keys[i] < k1);
    }
    for (size_t i = x1.size(); i < x1.size() + x2.size(); ++i)
    {
        EXPECT_TRUE(k1 <= keys[i] && keys[i] < k2);
    }
    for (size_t i = x1.size() + x2.size(); i < keys.size(); ++i)
    {
        EXPECT_TRUE(k2 <= keys[i]);
    }

    std::sort(keys.begin(), keys.end());
    auto uit = std::unique(keys.begin(), keys.end());

    // combined particles from duplicates should not contain any duplicate particles
    EXPECT_EQ(uit, keys.end());
}
