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
 * @brief Unit tests for ParticlesData
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <iostream>

#include "gtest/gtest.h"

#include "sph/particles_data.hpp"

using namespace sphexa;

TEST(ParticlesData, resize)
{
    ParticlesData<double, unsigned, cstone::CpuTag> d;

    d.setConserved("x", "y", "z");
    d.setDependent("du");

    size_t size = 10;
    d.resize(10);

    EXPECT_EQ(d.x.size(), size);
    EXPECT_EQ(d.y.size(), size);
    EXPECT_EQ(d.z.size(), size);
    EXPECT_EQ(d.du.size(), size);
}

TEST(ParticlesData, releaseAcquire)
{
    ParticlesData<double, unsigned, cstone::CpuTag> d;

    d.setConserved("x", "y", "z");
    d.setDependent("du", "ax", "p");

    size_t size = 10;
    d.resize(size);

    d.release("du");
    d.acquire("c");

    EXPECT_EQ(d.du.size(), 0);
    EXPECT_EQ(d.c.size(), size);
}

TEST(ParticlesData, releaseAcquireInt)
{
    ParticlesData<double, unsigned, cstone::CpuTag> d;

    d.setDependent("x");

    int indexOfX = std::find(d.fieldNames.begin(), d.fieldNames.end(), "x") - d.fieldNames.begin();

    size_t size = 10;
    d.resize(size);

    EXPECT_EQ(d.x.size(), size);
    EXPECT_EQ(d.y.size(), 0);

    d.release(indexOfX);
    d.acquire("y");

    EXPECT_EQ(d.x.size(), 0);
    EXPECT_EQ(d.y.size(), size);
}

TEST(ParticlesData, releaseAcquireMultiple)
{
    ParticlesData<double, unsigned, cstone::CpuTag> d;

    d.setDependent("du", "ax", "p");

    size_t size = 10;
    d.resize(size);

    EXPECT_EQ(d.du.size(), size);
    EXPECT_EQ(d.ax.size(), size);
    EXPECT_EQ(d.p.size(), size);

    d.release("du", "ax", "p");
    d.acquire("c11", "c12", "c13");

    EXPECT_EQ(d.du.size(), 0);
    EXPECT_EQ(d.ax.size(), 0);
    EXPECT_EQ(d.p.size(), 0);
    EXPECT_EQ(d.c11.size(), size);
    EXPECT_EQ(d.c12.size(), size);
    EXPECT_EQ(d.c13.size(), size);
}

TEST(ParticlesData, aquireThrow)
{
    ParticlesData<double, unsigned, cstone::CpuTag> d;

    d.setConserved("x", "y", "z");
    d.setDependent("du", "ax", "p");

    size_t size = 10;
    d.resize(size);

    // cannot acquire "c": no released field available
    EXPECT_ANY_THROW(d.acquire("c"));
}

TEST(ParticlesData, conservedThrow)
{
    ParticlesData<double, unsigned, cstone::CpuTag> d;

    d.setConserved("x", "y", "z");

    // cannot release "x" which is conserved
    EXPECT_ANY_THROW(d.release("x"));
}

TEST(ParticlesData, typeMismatch)
{
    ParticlesData<double, unsigned, cstone::CpuTag> d;

    d.setDependent("ax");
    d.resize(10);

    d.release("ax");

    // cannot acquire "nc" from released "x": types do not match
    EXPECT_ANY_THROW(d.acquire("nc"));
}
