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
 * @brief  Find neighbors in Morton code sorted x,y,z arrays
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "findneighbors.cuh"

template<class T, class Integer>
__global__ void findNeighborsKernel(const T* x, const T* y, const T* z, const T* h, int firstId, int lastId, int n,
                                    cstone::Box<T> box, const Integer* particleKeys,
                                    int* neighbors, int* neighborsCount, int ngmax)
{
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned id = firstId + tid;
    if (id < lastId)
    {
        cstone::findNeighbors(id, x, y, z, h, box, particleKeys, neighbors + tid*ngmax, neighborsCount + tid, n, ngmax);
    }
}

template<class T, class Integer>
void findNeighborsGpu(const T* x, const T* y, const T* z, const T* h, int firstId, int lastId, int n,
                       cstone::Box<T> box, const Integer* particleKeys, int* neighbors, int* neighborsCount, int ngmax,
                       cudaStream_t stream)
{
    unsigned numThreads = 256;
    unsigned numBlocks  = iceil(n, numThreads);
    findNeighborsKernel<<<numBlocks, numThreads, 0, stream>>>
        (x, y, z, h, firstId, lastId, n, box, particleKeys, neighbors, neighborsCount, ngmax);
}

template FIND_NEIGHBORS_GPU(float,  uint32_t)
template FIND_NEIGHBORS_GPU(float,  uint64_t)
template FIND_NEIGHBORS_GPU(double, uint32_t)
template FIND_NEIGHBORS_GPU(double, uint64_t)
