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

using namespace cstone;

template<class T, class KeyType>
__global__ void findNeighborsKernel(const T* x, const T* y, const T* z, const T* h, int firstId, int lastId, int n,
                                    cstone::Box<T> box, const KeyType* particleKeys,
                                    int* neighbors, int* neighborsCount, int ngmax)
{
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned id = firstId + tid;
    if (id >= lastId) { return; }

    findNeighbors(id, x, y, z, h, box, particleKeys, neighbors + tid*ngmax, neighborsCount + tid, n, ngmax);
}

//! @brief generic version
template<class T, class Integer>
inline
void findNeighborsGpu(const T* x, const T* y, const T* z, const T* h, int firstId, int lastId, int n,
                      cstone::Box<T> box, const Integer* particleKeys, int* neighbors, int* neighborsCount,
                      int ngmax, cudaStream_t stream)
{
    unsigned numThreads = 256;
    unsigned numBlocks  = iceil(n, numThreads);
    findNeighborsKernel<<<numBlocks, numThreads, 0, stream>>>
        (x, y, z, h, firstId, lastId, n, box, particleKeys, neighbors, neighborsCount, ngmax);
}

//! @brief Morton key based algorithm
template<class T, class Integer>
void findNeighborsMortonGpu(const T* x, const T* y, const T* z, const T* h, int firstId, int lastId, int n,
                            cstone::Box<T> box, const Integer* particleKeys, int* neighbors, int* neighborsCount,
                            int ngmax, cudaStream_t stream)
{
    const MortonKey<Integer>* mortonKeys = (MortonKey<Integer>*)(particleKeys);
    findNeighborsGpu(x, y, z, h, firstId, lastId, n, box, mortonKeys, neighbors, neighborsCount, ngmax, stream);
}

//! @brief Hilbert key based algorithm
template<class T, class Integer>
void findNeighborsHilbertGpu(const T* x, const T* y, const T* z, const T* h, int firstId, int lastId, int n,
                             cstone::Box<T> box, const Integer* particleKeys, int* neighbors, int* neighborsCount,
                             int ngmax, cudaStream_t stream)
{
    const HilbertKey<Integer>* hilbertKeys = (HilbertKey<Integer>*)(particleKeys);
    findNeighborsGpu(x, y, z, h, firstId, lastId, n, box, hilbertKeys, neighbors, neighborsCount, ngmax, stream);
}

template FIND_NEIGHBORS_MORTON_GPU(float,  uint32_t);
template FIND_NEIGHBORS_MORTON_GPU(float,  uint64_t);
template FIND_NEIGHBORS_MORTON_GPU(double, uint32_t);
template FIND_NEIGHBORS_MORTON_GPU(double, uint64_t);

template FIND_NEIGHBORS_HILBERT_GPU(float,  uint32_t);
template FIND_NEIGHBORS_HILBERT_GPU(float,  uint64_t);
template FIND_NEIGHBORS_HILBERT_GPU(double, uint32_t);
template FIND_NEIGHBORS_HILBERT_GPU(double, uint64_t);
