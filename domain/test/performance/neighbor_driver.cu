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

#include <iostream>
#include <iterator>

#include <thrust/device_vector.h>

#include "cstone/findneighbors.hpp"
#include "cstone/cuda/findneighbors.cuh"

#include "../coord_samples/random.hpp"
#include "timing.cuh"

using namespace cstone;

template<class T, class KeyType>
void benchmarkGpu()
{
    using Integer = typename KeyType::ValueType;

    Box<T> box{0, 1, true};
    int n = 2000000;

    RandomCoordinates<T, KeyType> coords(n, box);
    std::vector<T> h(n, 0.012);

    int ngmax = 200;
    std::vector<int> neighborsGPU(ngmax * n);
    std::vector<int> neighborsCountGPU(n);

    thrust::device_vector<T> d_x(coords.x().begin(), coords.x().end());
    thrust::device_vector<T> d_y(coords.y().begin(), coords.y().end());
    thrust::device_vector<T> d_z(coords.z().begin(), coords.z().end());
    thrust::device_vector<T> d_h = h;
    thrust::device_vector<Integer> d_codes(coords.particleKeys().begin(), coords.particleKeys().end());

    thrust::device_vector<int> d_neighbors(neighborsGPU.size());
    thrust::device_vector<int> d_neighborsCount(neighborsCountGPU.size());

    const auto* deviceKeys = (const KeyType*)(thrust::raw_pointer_cast(d_codes.data()));

    auto findNeighborsLambda = [&]()
    {
      findNeighborsSfcGpu(
          thrust::raw_pointer_cast(d_x.data()), thrust::raw_pointer_cast(d_y.data()),
          thrust::raw_pointer_cast(d_z.data()), thrust::raw_pointer_cast(d_h.data()), 0, n, n, box,
          deviceKeys, thrust::raw_pointer_cast(d_neighbors.data()),
          thrust::raw_pointer_cast(d_neighborsCount.data()), ngmax);
    };

    float gpuTime = timeGpu(findNeighborsLambda);

    thrust::copy(d_neighborsCount.begin(), d_neighborsCount.end(), neighborsCountGPU.begin());

    std::cout << "GPU time " << gpuTime / 1000 << " s" << std::endl;
    std::copy(neighborsCountGPU.data(), neighborsCountGPU.data() + 10, std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;

    std::vector<int> neighborsCPU(ngmax * n);
    std::vector<int> neighborsCountCPU(n);

    const T* x = coords.x().data();
    const T* y = coords.y().data();
    const T* z = coords.z().data();
    const auto* codes = (KeyType*)(coords.particleKeys().data());

    auto findNeighborsCpu = [&]()
    {
        findNeighbors(x, y, z, h.data(), 0, n, n, box, codes, neighborsCPU.data(), neighborsCountCPU.data(), ngmax);
    };

    float cpuTime = timeCpu(findNeighborsCpu);

    std::cout << "CPU time " << cpuTime << " s" << std::endl;
    std::copy(neighborsCountCPU.data(), neighborsCountCPU.data() + 10, std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;

    bool allEqual = std::equal(begin(neighborsCountGPU), end(neighborsCountGPU), begin(neighborsCountCPU));
    if (allEqual)
        std::cout << "Neighbor counts: PASS\n";
    else
        std::cout << "Neighbor counts: FAIL\n";

}

int main()
{
    benchmarkGpu<double, MortonKey<uint64_t>>();
    benchmarkGpu<double, HilbertKey<uint64_t>>();
}
