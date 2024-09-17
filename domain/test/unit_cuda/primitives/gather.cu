
/*! @file
 * @brief GPU SFC sorter unit tests
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 */

#include <vector>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include "gtest/gtest.h"

#include "cstone/cuda/thrust_util.cuh"
#include "cstone/primitives/gather.cuh"

using namespace cstone;

TEST(SfcSorterGpu, shiftMapLeft)
{
    using KeyType   = unsigned;
    using IndexType = unsigned;
    using thrust::raw_pointer_cast;

    thrust::device_vector<KeyType> keys = std::vector<KeyType>{2, 1, 5, 4};

    thrust::device_vector<IndexType> obuf, keyBuf, valBuf;
    GpuSfcSorter<IndexType, thrust::device_vector<unsigned>> sorter(obuf);

    sorter.setMapFromCodes(raw_pointer_cast(keys.data()), raw_pointer_cast(keys.data()) + keys.size(), keyBuf, valBuf);
    // map is [1 0 3 2]

    {
        thrust::device_vector<IndexType> ref = std::vector<IndexType>{1, 0, 3, 2};
        EXPECT_EQ(obuf, ref);
    }

    sorter.extendMap(-1, keyBuf);

    {
        thrust::device_vector<IndexType> ref = std::vector<IndexType>{0, 2, 1, 4, 3};
        EXPECT_EQ(obuf, ref);
    }
}

TEST(SfcSorterGpu, shiftMapRight)
{
    using KeyType   = unsigned;
    using IndexType = unsigned;
    using thrust::raw_pointer_cast;

    thrust::device_vector<KeyType> keys = std::vector<KeyType>{2, 1, 5, 4};

    thrust::device_vector<IndexType> obuf, keyBuf, valBuf;
    GpuSfcSorter<IndexType, thrust::device_vector<unsigned>> sorter(obuf);

    sorter.setMapFromCodes(raw_pointer_cast(keys.data()), raw_pointer_cast(keys.data()) + keys.size(), keyBuf, valBuf);
    // map is [1 0 3 2]

    sorter.extendMap(1, keyBuf);
    {
        thrust::device_vector<IndexType> ref = std::vector<IndexType>{1, 0, 3, 2, 4};
        EXPECT_EQ(obuf, ref);
    }
}
