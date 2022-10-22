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
 * @brief Utility functions.
 * Mostly stolen from jolatechno/QuIDS (https://github.com/jolatechno/QuIDS).
 *
 * @author Joseph Touzet <joseph.touzet@ens-paris-saclay.fr>
 */

#pragma once

#include <vector>
#include <parallel/algorithm>
#include <parallel/numeric>

#include <omp.h>

namespace util
{
/*! @brief parallel iota */
template<class iteratorType, class valueType>
void parallel_iota(iteratorType begin, iteratorType end, const valueType value_begin)
{
    size_t distance = std::distance(begin, end);

    if (value_begin == 0)
    {
#pragma omp parallel for
        for (size_t i = 0; i < distance; ++i)
            begin[i] = i;
    }
    else
#pragma omp parallel for
        for (size_t i = 0; i < distance; ++i)
            begin[i] = value_begin + i;
}

/*! @brief linear partitioning algorithm into n partitions without an initialized index list in parallel */
template<class idIteratorType, class countIteratorType, class functionType>
void parallel_generalized_partition_from_iota(idIteratorType idx_in, idIteratorType idx_in_end,
                                              long long int const iotaOffset, countIteratorType offset,
                                              countIteratorType offset_end, functionType const partitioner)
{

    int const           n_segment = std::distance(offset, offset_end) - 1;
    long long int const id_end    = std::distance(idx_in, idx_in_end);

    // limit values
    offset[0]         = 0;
    offset[n_segment] = id_end;

    if (n_segment == 1)
    {
        parallel_iota(idx_in, idx_in_end, iotaOffset);
        return;
    }
    if (id_end == 0)
    {
        std::fill(offset, offset_end, 0);
        return;
    }

    // number of threads
    int num_threads;
#pragma omp parallel
#pragma omp single
    num_threads = omp_get_num_threads();

    std::vector<size_t> count(n_segment * num_threads, 0);

#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();

        long long int begin = id_end * thread_id / num_threads;
        long long int end   = id_end * (thread_id + 1) / num_threads;
        for (long long int i = end + iotaOffset - 1; i >= begin + iotaOffset; --i)
        {
            auto key = partitioner(i);
            ++count[key * num_threads + thread_id];
        }
    }

    __gnu_parallel::partial_sum(count.begin(), count.begin() + n_segment * num_threads, count.begin());

#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();

        long long int begin = id_end * thread_id / num_threads;
        long long int end   = id_end * (thread_id + 1) / num_threads;
        for (long long int i = begin + iotaOffset; i < end + iotaOffset; ++i)
        {
            auto key                                       = partitioner(i);
            idx_in[--count[key * num_threads + thread_id]] = i;
        }
    }

#pragma omp parallel for
    for (int i = 1; i < n_segment; ++i)
        offset[i] = count[i * num_threads];
}
} // namespace util
