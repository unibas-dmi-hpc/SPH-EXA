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
 * @brief Utility tests
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <numeric>
#include <vector>
#include "gtest/gtest.h"

#include "cstone/util/aligned_alloc.hpp"
#include "cstone/util/noinit_alloc.hpp"

using namespace util;

template<class T>
using AllocatorType = DefaultInitAdaptor<T, AlignedAllocator<T, 64>>;
//using AllocatorType = DefaultInitAdaptor<T, std::allocator<T>>;

/*! @brief noinit allocation test
 *
 * Heap memory pages do get initialized to zero by the OS the first time
 * they are given to the process.
 * Therefore, to see any effects, i.e. vectors with non-zero elements after construction,
 * we first have to allocate a couple of pages and fill them with non-zero data.
 * Then we stand a good change that the final vector will get a page previously touched
 * by the process, which will contain non-zero data from previous allocations.
 */
TEST(Utils, AlignedNoInitAlloc)
{
    for (int i = 0; i < 100; ++i)
    {
        std::vector<double, AllocatorType<double>> v(512);
        std::iota(v.begin(), v.end(), 0);
    }

    // most likely allocated with previously touched pages
    std::vector<double, AllocatorType<double>> v(10);

    // content of v is uninitialized (not all zeros)
    EXPECT_NE(std::count(v.begin(), v.end(), 0.0), v.size());

    // address is 64-byte aligned
    EXPECT_EQ(reinterpret_cast<size_t>(v.data()) % 64, 0);

    // init to zero still works if explicitly specified
    std::vector<double, AllocatorType<double>> z(10, 0);
    EXPECT_EQ(std::count(z.begin(), z.end(), 0.0), z.size());
    EXPECT_EQ(reinterpret_cast<size_t>(z.data()) % 64, 0);

    //std::copy(v.begin(), v.end(), std::ostream_iterator<double>(std::cout, " "));
    //std::cout << " address: " << v.data() << std::endl;
}
