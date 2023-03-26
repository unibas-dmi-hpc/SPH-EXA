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
 * @brief Cornerstone octree GPU testing
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 */

#include "gtest/gtest.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "cstone/traversal/groups.cuh"
#include "cstone/util/util.hpp"
#include "cstone/cuda/cuda_utils.cuh"

using namespace cstone;

TEST(TargetGroups, t0)
{
    using T = double;
    LocalIndex numGroups = 4;
    LocalIndex groupSize = 8;
    LocalIndex first = 4;
    LocalIndex last = 34;

    thrust::device_vector<LocalIndex> groups(numGroups + 1);

    groupTargets<T><<<iceil(last - first, 64), 64>>>(first, last, nullptr, nullptr, nullptr, nullptr, groupSize,
                                                     rawPtr(groups), iceil(last - first, groupSize));

    thrust::host_vector<LocalIndex> hgroups = groups;

    thrust::host_vector<LocalIndex> ref = std::vector<LocalIndex>{4,12,20,28,34};

    EXPECT_EQ(hgroups, ref);
}
