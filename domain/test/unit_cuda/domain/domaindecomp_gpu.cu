/*
 * MIT License
 *
 * Copyright (c) 2022 CSCS, ETH Zurich
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

#include <thrust/device_vector.h>

#include "gtest/gtest.h"

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/domain/domaindecomp_gpu.cuh"

using namespace cstone;

/*! @brief test SendList creation from a SFC assignment
 *
 * This test creates an array with SFC keys and an
 * SFC assignment with SFC keys ranges.
 * CreateSendList then translates the code ranges into indices
 * valid for the SFC key array.
 */
template<class KeyType>
static void sendListMinimalGpu()
{
    std::vector<KeyType> tree{0, 2, 6, 8, 10};
    std::vector<KeyType> codes{0, 0, 1, 3, 4, 5, 6, 6, 9};

    int numRanks = 2;
    SpaceCurveAssignment assignment(numRanks);
    assignment.addRange(0, 0, 2, 0);
    assignment.addRange(1, 2, 4, 0);

    thrust::device_vector<KeyType> d_keys = codes;
    thrust::device_vector<KeyType> d_searchKeys(numRanks);
    thrust::device_vector<LocalIndex> d_indices(numRanks);

    gsl::span<const KeyType> d_keyView{rawPtr(d_keys), d_keys.size()};

    // note: codes input needs to be sorted
    auto sendList = createSendRangesGpu<KeyType>(assignment, tree, d_keyView, rawPtr(d_searchKeys), rawPtr(d_indices));

    EXPECT_EQ(sendList.count(0), 6);
    EXPECT_EQ(sendList.count(1), 3);

    EXPECT_EQ(sendList[0], 0);
    EXPECT_EQ(sendList[1], 6);
}

TEST(DomainDecomposition, createSendListGpu)
{
    sendListMinimalGpu<unsigned>();
    sendListMinimalGpu<uint64_t>();
}