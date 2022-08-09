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
 * @brief  Utility for GPU-direct domain particle exchange
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gather_gpu.hpp"

#include "cstone/util/util.hpp"

namespace cstone
{

template<class T, class IndexType>
__global__ void gatherGpuKernel(const IndexType* map, size_t n, const T* source, T* destination)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) { destination[tid] = source[map[tid]]; }
}

template<class T, class IndexType>
void gatherGpu(const IndexType* map, size_t n, const T* source, T* destination)
{
    int numThreads = 256;
    int numBlocks  = iceil(n, numThreads);

    gatherGpuKernel<<<numBlocks, numThreads>>>(map, n, source, destination);
}

template void gatherGpu(const unsigned*, size_t, const int*, int*);
template void gatherGpu(const unsigned*, size_t, const util::array<float, 1>*, util::array<float, 1>*);
template void gatherGpu(const unsigned*, size_t, const util::array<float, 2>*, util::array<float, 2>*);
template void gatherGpu(const unsigned*, size_t, const util::array<float, 3>*, util::array<float, 3>*);
template void gatherGpu(const unsigned*, size_t, const util::array<float, 4>*, util::array<float, 4>*);

} // namespace cstone