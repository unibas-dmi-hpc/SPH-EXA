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
 * @brief Benchmark cornerstone octree generation on the GPU
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>

#include "cstone/sfc/hilbert.hpp"

using namespace cstone;

int main()
{
    using KeyType = uint64_t;
    unsigned numKeys = 32000000;

    int maxCoord = (1 << maxTreeLevel<KeyType>{}) - 1;
    std::mt19937 gen;
    std::uniform_int_distribution<unsigned> distribution(0, maxCoord);
    auto getRand = [&distribution, &gen](){ return distribution(gen); };

    std::vector<unsigned> x(numKeys);
    std::vector<unsigned> y(numKeys);
    std::vector<unsigned> z(numKeys);

    std::generate(begin(x), end(x), getRand);
    std::generate(begin(y), end(y), getRand);
    std::generate(begin(z), end(z), getRand);

    std::vector<KeyType> sfcKeys(numKeys);

    auto cpu_t0 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numKeys; ++i)
    {
        sfcKeys[i] = iHilbert<KeyType>(x[i], y[i], z[i]);
    }
    auto cpu_t1 = std::chrono::high_resolution_clock::now();
    double cpu_time_hilbert = std::chrono::duration<double>(cpu_t1 - cpu_t0).count();

    std::cout << "compute time for " << numKeys << " hilbert keys: " << cpu_time_hilbert
              << " s on CPU" << std::endl;

    cpu_t0 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numKeys; ++i)
    {
        sfcKeys[i] = imorton3D<KeyType>(x[i], y[i], z[i]);
    }
    cpu_t1 = std::chrono::high_resolution_clock::now();
    double cpu_time_morton = std::chrono::duration<double>(cpu_t1 - cpu_t0).count();

    std::cout << "compute time for " << numKeys << " morton keys: " << cpu_time_morton
              << " s on CPU" << std::endl;
}
