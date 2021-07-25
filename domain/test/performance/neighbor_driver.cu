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

#include <thrust/device_vector.h>

#include "cstone/findneighbors.hpp"
#include "cstone/cuda/findneighbors.cuh"

#include "../coord_samples/random.hpp"
#include "timing.cuh"

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
    const CodeType* codes = coords.particleKeys().data();

    thrust::device_vector<T> d_x = coords.x();
    thrust::device_vector<T> d_y = coords.y();
    thrust::device_vector<T> d_z = coords.z();
    thrust::device_vector<T> d_h = h;
    thrust::device_vector<CodeType> d_codes = coords.particleKeys();

    thrust::device_vector<int> d_neighbors(neighborsGPU.size());
    thrust::device_vector<int> d_neighborsCount(neighborsCountGPU.size());

    auto findNeighborsLambda = [&]()
    {
        findNeighborsCuda(thrust::raw_pointer_cast(d_x.data()), thrust::raw_pointer_cast(d_y.data()),
                          thrust::raw_pointer_cast(d_z.data()), thrust::raw_pointer_cast(d_h.data()), 0, n, n, box,
                          thrust::raw_pointer_cast(d_codes.data()), thrust::raw_pointer_cast(d_neighbors.data()),
                          thrust::raw_pointer_cast(d_neighborsCount.data()), ngmax);
    };

    float gpuTime = timeGpu(findNeighborsLambda);

    thrust::copy(d_neighborsCount.begin(), d_neighborsCount.end(), neighborsCountGPU.begin());

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
}
