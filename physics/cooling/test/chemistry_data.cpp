/*
 * MIT License
 *
 * Copyright (c) 2022 CSCS, ETH Zurich, University of Zurich, University of Basel
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
 * @brief Radiative cooling tests with GRACKLE
 *
 * @author Noah Kubli <noah.kubli@uzh.ch>
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <iostream>
#include <vector>

#include "gtest/gtest.h"

#include "cooling/chemistry_data.hpp"

using namespace cooling;
using cstone::get;

TEST(ChemistryData, test1a)
{
    using T = double;
    ChemistryData<T> data;

    // activate some of the fields at runtime, affects next resize
    data.setConserved(0, 1, 9);

    size_t dataSize = 10;
    data.resize(dataSize);

    EXPECT_EQ(data.fields[0].size(), dataSize);
    EXPECT_EQ(data.fields[1].size(), dataSize);
    EXPECT_EQ(data.fields[9].size(), dataSize);

    EXPECT_EQ(data.fields[2].size(), 0);

    // fields can also be accessed based on names
    EXPECT_EQ(get<"Y0">(data).data(), data.fields[0].data());
}
