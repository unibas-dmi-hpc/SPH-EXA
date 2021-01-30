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
 * \brief  Find neighbors in Morton code sorted x,y,z arrays
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <iostream>
#include <iterator>

#include "../coord_samples/random.hpp"
#include "../../include/cstone/findneighbors.hpp"


template<class T, class I>
__global__ void findNeighborsCuda(const T* x, const T* y, const T* z, const T* h, int firstId, int lastId, int n,
                                  Box<T> box, const I* codes, int* neighbors, int* neighborsCount, int ngmax)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int id = firstId + tid;
    if (id < lastId)
    {
        cstone::findNeighbors(id, x, y, z, h, box, codes, neighbors + tid*ngmax, neighborsCount + tid, n, ngmax);
    }
}

int main()
{
    using CodeType = unsigned;
    using T        = float;

    Box<T> box{0,1, true};
    int n = 2000000;

    RandomCoordinates<T, CodeType> coords(n, box);
    std::vector<T> h(n, 0.006);

    int ngmax = 100;
    std::vector<int> neighbors(ngmax * n);
    std::vector<int> neighborsCount(n);

    const T* x = coords.x().data();
    const T* y = coords.y().data();
    const T* z = coords.z().data();
    const CodeType* codes = coords.mortonCodes().data();

    int* neighs = neighbors.data();
    int* neighsC = neighborsCount.data();

    T* d_x;
    T* d_y;
    T* d_z;
    T* d_h;
    CodeType* d_codes;
    int* d_neighs;
    int* d_neighsC;

    cudaMalloc((void **)&d_x, sizeof(T) * n);
    cudaMalloc((void **)&d_y, sizeof(T) * n);
    cudaMalloc((void **)&d_z, sizeof(T) * n);
    cudaMalloc((void **)&d_h, sizeof(T) * n);
    cudaMalloc((void **)&d_codes, sizeof(T) * n);
    cudaMalloc((void **)&d_neighs, sizeof(int) * neighbors.size());
    cudaMalloc((void **)&d_neighsC, sizeof(int) * neighborsCount.size());

    cudaMemcpy(d_x, x, sizeof(T) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, sizeof(T) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, z, sizeof(T) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_h, h.data(), sizeof(T) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_codes, codes, sizeof(T) * n, cudaMemcpyHostToDevice);

    findNeighborsCuda<<<1, 10>>>(d_x, d_y, d_z, d_h, 0, n, n, box, d_codes, d_neighs, d_neighsC, ngmax);
    cudaMemcpy(neighsC, d_neighsC, n * sizeof(int), cudaMemcpyDeviceToHost);

    std::copy(neighsC, neighsC + 10, std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;

    #pragma omp parallel for
    for (int id = 0; id < 10; ++id)
    {
        cstone::findNeighbors(id, x, y, z, h.data(), box, codes, neighs + id*ngmax, neighsC + id, n, ngmax);
    }
    std::copy(neighsC, neighsC + 10, std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_h);
    cudaFree(d_neighs);
    cudaFree(d_neighsC);
}
