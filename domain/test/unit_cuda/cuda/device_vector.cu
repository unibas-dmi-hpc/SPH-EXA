/*! @file
 * @brief  Tests for SFC related GPU device functions
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <numeric>
#include <vector>

#include "gtest/gtest.h"

#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/cuda/device_vector.h"
#include "cstone/util/reallocate.hpp"

using namespace cstone;

TEST(CudaUtils, DeviceVector)
{
    DeviceVector<int> v;

    std::size_t initSize = 10;
    v.resize(initSize);

    EXPECT_EQ(v.size(), initSize);
    EXPECT_EQ(v.capacity(), initSize);

    thrust::sequence(thrust::device, v.data(), v.data() + v.size(), 0);

    double growthFactor = 1.5;
    reallocate(v, 2 * initSize, growthFactor);

    EXPECT_EQ(v.size(), 2 * initSize);
    EXPECT_EQ(v.capacity(), 2 * initSize * growthFactor);

    thrust::sequence(thrust::device, v.data() + initSize, v.data() + v.size(), initSize);

    std::vector<int> h_v(v.size());
    memcpyD2H(v.data(), v.size(), h_v.data());

    std::vector<int> ref(v.size());
    std::iota(ref.begin(), ref.end(), 0);

    EXPECT_EQ(ref, h_v);
}
