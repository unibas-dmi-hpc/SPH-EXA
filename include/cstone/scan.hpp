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

/*! \file
 * \brief Parallel prefix sum
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <algorithm>
#include <iostream>

#include <omp.h>

namespace cstone
{

template<class T>
void exclusiveScan(const T* in, T* out, std::size_t numElements)
{
    constexpr int blockSize = 16384 / sizeof(T);

    int numThreads = 0;
    #pragma omp parallel
    {
        #pragma omp single
        numThreads = omp_get_num_threads();
    }
    std::cout << "numThreads is: " << numThreads << std::endl;

    T superBlock[numThreads];
    std::fill(superBlock, superBlock + numThreads, 0);

    unsigned elementsPerStep = numThreads * blockSize;
    unsigned nSteps = numElements / elementsPerStep;

    size_t stepSum = 0;

    #pragma omp parallel num_threads(numThreads)
    {
        int tid = omp_get_thread_num();
        for (std::size_t step = 0; step < nSteps; ++step)
        {
            //#pragma omp parallel num_threads(numThreads)
            //for (int tid = 0; tid < numThreads; ++tid)
            {
                //int tid = omp_get_thread_num();
                size_t stepOffset = step * elementsPerStep + tid * blockSize;

                T tSum = 0;
                for (std::size_t ib = 0; ib < blockSize; ++ib)
                {
                    size_t i = stepOffset + ib;

                    out[i] = tSum + stepSum;
                    tSum += in[i];
                }

                superBlock[tid] = tSum;
            }

            #pragma omp barrier

            if (tid == 0)
            {
                // inclusive scan the super block
                for (int blockId = 1; blockId < numThreads; ++blockId)
                {
                    superBlock[blockId] += superBlock[blockId - 1];
                }
            }

            #pragma omp barrier

            //#pragma omp parallel num_threads(numThreads)
            //for (int tid = 1; tid < numThreads; ++tid)
            {
                //int tid = omp_get_thread_num();
                if (tid > 0)
                {
                    size_t stepOffset = step * elementsPerStep + tid * blockSize;

                    T superBlockSum = superBlock[tid - 1];
                    for (std::size_t ib = 0; ib < blockSize; ++ib)
                    {
                        size_t i = stepOffset + ib;
                        out[i] += superBlockSum;
                    }
                }
            }

            if (tid == 0)
            {
                stepSum += superBlock[numThreads - 1];
            }
            #pragma omp barrier
        }
    }

    for (size_t i = nSteps * elementsPerStep; i < numElements; ++i)
    {
        out[i] = stepSum;
        stepSum += in[i];
    }
}

} // namespace cstone
