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

#include <chrono>
#include <iostream>
#include <iterator>

#include <cuda_runtime.h>

#include "../coord_samples/random.hpp"
#include "cstone/findneighbors.hpp"

#include "cstone/cuda/findneighbors.cuh"

int main()
{
    using CodeType = unsigned;
    using T = float;

    Box<T> box{0, 1, true};
    int n = 2000000;

    RandomCoordinates<T, CodeType> coords(n, box);
    std::vector<T> h(n, 0.006);

    int ngmax = 100;
    std::vector<int> neighborsGPU(ngmax * n);
    std::vector<int> neighborsCountGPU(n);

    const T* x = coords.x().data();
    const T* y = coords.y().data();
    const T* z = coords.z().data();
    const CodeType* codes = coords.mortonCodes().data();

    T* d_x;
    T* d_y;
    T* d_z;
    T* d_h;
    CodeType* d_codes;
    int* d_neighbors;
    int* d_neighborsCount;

    cudaMalloc((void**)&d_x, sizeof(T) * n);
    cudaMalloc((void**)&d_y, sizeof(T) * n);
    cudaMalloc((void**)&d_z, sizeof(T) * n);
    cudaMalloc((void**)&d_h, sizeof(T) * n);
    cudaMalloc((void**)&d_codes, sizeof(T) * n);
    cudaMalloc((void**)&d_neighbors, sizeof(int) * neighborsGPU.size());
    cudaMalloc((void**)&d_neighborsCount, sizeof(int) * neighborsCountGPU.size());

    cudaMemcpy(d_x, x, sizeof(T) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, sizeof(T) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, z, sizeof(T) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_h, h.data(), sizeof(T) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_codes, codes, sizeof(T) * n, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, cudaStreamDefault);
    findNeighborsCuda(d_x, d_y, d_z, d_h, 0, n, n, box, d_codes, d_neighbors, d_neighborsCount, ngmax);
    cudaEventRecord(stop, cudaStreamDefault);
    cudaEventSynchronize(stop);

    cudaMemcpy(neighborsCountGPU.data(), d_neighborsCount, n * sizeof(int), cudaMemcpyDeviceToHost);

    float gpuTime;
    cudaEventElapsedTime(&gpuTime, start, stop);

    std::cout << "GPU time " << gpuTime / 1000 << " s" << std::endl;
    std::copy(neighborsCountGPU.data(), neighborsCountGPU.data() + 10, std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;

    std::vector<int> neighborsCPU(ngmax * n);
    std::vector<int> neighborsCountCPU(n);

    auto t0 = std::chrono::high_resolution_clock::now();

#pragma omp parallel for
    for (int id = 0; id < n; ++id)
    {
        cstone::findNeighbors(id, x, y, z, h.data(), box, codes, neighborsCPU.data() + id * ngmax,
                              neighborsCountCPU.data() + id, n, ngmax);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double cpuTime = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "CPU time " << cpuTime << " s" << std::endl;
    std::copy(neighborsCountCPU.data(), neighborsCountCPU.data() + 10, std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;

    bool allEqual = std::equal(begin(neighborsCountGPU), end(neighborsCountGPU), begin(neighborsCountCPU));
    if (allEqual)
        std::cout << "Neighbor counts: PASS\n";
    else
        std::cout << "Neighbor counts: FAIL\n";

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_h);
    cudaFree(d_neighbors);
    cudaFree(d_neighborsCount);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
