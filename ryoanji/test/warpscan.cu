
#include "gtest/gtest.h"

#include "ryoanji/types.h"
#include "ryoanji/warpscan.h"

__global__ void testScan(int* values)
{
    int val = 1;
    int scan = inclusiveScanInt(val);
    values[threadIdx.x] = scan;
}


TEST(WarpScan, inclusiveInt)
{
    cudaVec<int> values(64, true);
    testScan<<<1,64>>>(values.d());
    values.d2h();

    for (int i = 0; i < 64; ++i)
    {
        EXPECT_EQ(values[i], i % WARP_SIZE + 1);
    }
}

__global__ void testScanBool(int* result)
{
    bool val = threadIdx.x % 2;
    result[threadIdx.x] = exclusiveScanBool(val);
}

TEST(WarpScan, bools)
{
    cudaVec<int> values(64, true);
    testScanBool<<<1,64>>>(values.d());
    values.d2h();

    for (int i = 0; i < 64; ++i)
    {
        EXPECT_EQ(values[i], (i % WARP_SIZE) / 2);
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
    cudaVec<int> values(32, true);
    testSegScan<<<1,32>>>(values.d());
    values.d2h();

    //                         carry is one, first segment starts with offset of 1
    //                         |                                         | value(16) = -2, so scan restarts at 2 - 1
    std::vector<int> reference{2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18, 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,2};
    //                                                              value(31) = -3, scan restarts at 3 - 1  ^

    for (int i = 0; i < 32; ++i)
    {
        EXPECT_EQ(values[i], reference[i]);
    }
}

