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
 * @brief  Thrust sorting and prefix sums
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <thrust/scan.h>
#include <thrust/sort.h>

#include "cstone/cuda/annotation.hpp"
#include "cstone/cuda/thrust_sort_scan.cuh"
#include "cstone/sfc/common.hpp"


void thrust_exclusive_scan(const cstone::TreeNodeIndex* first, const cstone::TreeNodeIndex* last,
                           cstone::TreeNodeIndex* dest)
{
    thrust::exclusive_scan(thrust::device, first, last, dest);
}

void thrust_exclusive_scan(const unsigned* first, const unsigned* last, unsigned* dest)
{
    thrust::exclusive_scan(thrust::device, first, last, dest);
}

void thrust_sort_by_key(cstone::TreeNodeIndex* firstKey, cstone::TreeNodeIndex* lastKey, cstone::TreeNodeIndex* value)
{
    thrust::sort_by_key(thrust::device, firstKey, lastKey, value);
}

/*! @brief functor to sort octree nodes first according to level, then by SFC key
 *
 * Note: takes SFC keys with Warren-Salmon placeholder bits in place as arguments
 */
template<class KeyType>
struct compareLevelThenPrefix
{
    HOST_DEVICE_FUN bool operator()(KeyType a, KeyType b) const
    {
        unsigned prefix_a = cstone::decodePrefixLength(a);
        unsigned prefix_b = cstone::decodePrefixLength(b);

        if (prefix_a < prefix_b)
        {
            return true;
        }
        else if (prefix_b < prefix_a)
        {
            return false;
        }
        else
        {
            return cstone::decodePlaceholderBit(a) < cstone::decodePlaceholderBit(b);
        }
    }
};

void thrust_sort_by_level_and_key(uint32_t* firstKey, uint32_t* lastKey, cstone::TreeNodeIndex* value)
{
    thrust::sort_by_key(thrust::device, firstKey, lastKey, value, compareLevelThenPrefix<uint32_t>{});
}

void thrust_sort_by_level_and_key(uint64_t* firstKey, uint64_t* lastKey, cstone::TreeNodeIndex* value)
{
    thrust::sort_by_key(thrust::device, firstKey, lastKey, value, compareLevelThenPrefix<uint64_t>{});
}
