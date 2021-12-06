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
 * @brief  Tests for warp-level primitives
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "ryoanji/types.h"
#include "ryoanji/warpscan.cuh"

using namespace ryoanji;

__global__ void testMin(int* values)
{
    int laneValue = threadIdx.x;

    values[threadIdx.x] = warpMin(laneValue);
}

TEST(WarpScan, min)
{
    thrust::host_vector<int>   h_v(GpuConfig::warpSize);
    thrust::device_vector<int> d_v = h_v;

    testMin<<<1, GpuConfig::warpSize>>>(thrust::raw_pointer_cast(d_v.data()));

    h_v = d_v;
    thrust::host_vector<int> reference(GpuConfig::warpSize, 0);

    EXPECT_EQ(h_v, reference);
}

__global__ void testMax(int* values)
{
    int laneValue = threadIdx.x;

    values[threadIdx.x] = warpMax(laneValue);
}

TEST(WarpScan, max)
{
    thrust::host_vector<int>   h_v(GpuConfig::warpSize);
    thrust::device_vector<int> d_v = h_v;

    testMax<<<1, GpuConfig::warpSize>>>(thrust::raw_pointer_cast(d_v.data()));

    h_v = d_v;
    thrust::host_vector<int> reference(GpuConfig::warpSize, GpuConfig::warpSize - 1);

    EXPECT_EQ(h_v, reference);
}

__global__ void testScan(int* values)
{
    int val = 1;
    int scan = inclusiveScanInt(val);
    values[threadIdx.x] = scan;
}


TEST(WarpScan, inclusiveInt)
{
    thrust::device_vector<int> d_values(2 * GpuConfig::warpSize);
    testScan<<<1, 2 * GpuConfig::warpSize>>>(rawPtr(d_values.data()));
    thrust::host_vector<int> h_values = d_values;

    for (int i = 0; i < 2 * GpuConfig::warpSize; ++i)
    {
        EXPECT_EQ(h_values[i], i % GpuConfig::warpSize + 1);
    }
}

__global__ void testScanBool(int* result)
{
    bool val = threadIdx.x % 2;
    result[threadIdx.x] = exclusiveScanBool(val);
}

TEST(WarpScan, bools)
{
    thrust::device_vector<int> d_values(2 * GpuConfig::warpSize);
    testScanBool<<<1, 2 * GpuConfig::warpSize>>>(rawPtr(d_values.data()));
    thrust::host_vector<int> h_values = d_values;

    for (int i = 0; i < 2 * GpuConfig::warpSize; ++i)
    {
        EXPECT_EQ(h_values[i], (i % GpuConfig::warpSize) / 2);
    }
}

__global__ void testSegScan(int* values)
{
    int val = 1;

    if (threadIdx.x == 8)
        val = 2;

    if (threadIdx.x == 16)
        val = -2;

    if (threadIdx.x == 31)
        val = -3;

    int carry = 1;
    int scan = inclusiveSegscanInt(val, carry);
    values[threadIdx.x] = scan;
}

TEST(WarpScan, inclusiveSegInt)
{
    thrust::device_vector<int> d_values(32);
    testSegScan<<<1, 32>>>(rawPtr(d_values.data()));
    thrust::host_vector<int> h_values = d_values;

    //                         carry is one, first segment starts with offset of 1
    //                         |                                        | value(16) = -2, so scan restarts at 2 - 1
    std::vector<int> reference{2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18,
                               1, 2, 3, 4, 5, 6, 7, 8, 9,  10, 11, 12, 13, 14, 15, 2};
    //                                                              value(31) = -3, scan restarts at 3 - 1  ^

    for (int i = 0; i < 32; ++i)
    {
        EXPECT_EQ(h_values[i], reference[i]);
    }
}
