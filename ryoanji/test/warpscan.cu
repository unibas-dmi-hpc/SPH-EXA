
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
    //result[threadIdx.x] = lanemask_lt();
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
