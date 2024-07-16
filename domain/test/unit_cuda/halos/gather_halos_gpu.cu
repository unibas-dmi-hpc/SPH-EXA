/*
 * MIT License
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *               2024 University of Basel
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
 * @brief Halo exchange auxiliary functions GPU testing
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 */

#include "gtest/gtest.h"

#include <numeric>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "cstone/cuda/thrust_util.cuh"
#include "cstone/halos/gather_halos_gpu.h"

using namespace cstone;

TEST(Halos, gatherRanges)
{
    // list of marked halo cells/ranges
    std::vector<int> seq(30);
    std::iota(seq.begin(), seq.end(), 0);
    thrust::device_vector<int> src = seq;

    thrust::device_vector<unsigned> rangeScan    = std::vector<unsigned>{0, 4, 7};
    thrust::device_vector<unsigned> rangeOffsets = std::vector<unsigned>{4, 12, 22};
    int totalCount                               = 10;

    thrust::device_vector<int> buffer = std::vector<int>(totalCount);

    gatherRanges(rawPtr(rangeScan), rawPtr(rangeOffsets), rangeScan.size(), rawPtr(src), rawPtr(buffer), totalCount);

    thrust::host_vector<int> h_buffer = buffer;
    thrust::host_vector<int> ref      = std::vector<int>{4, 5, 6, 7, 12, 13, 14, 22, 23, 24};

    EXPECT_EQ(h_buffer, ref);
}
