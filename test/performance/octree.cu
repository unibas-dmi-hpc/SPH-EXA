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

#include <chrono>
#include <iostream>

#include <thrust/reduce.h>

#include "cstone/tree/octree.cuh"

#include "coord_samples/random.hpp"

using namespace cstone;

int main()
{
    using CodeType = unsigned;
    Box<double> box{-1, 1};

    int nParticles = 2000000;
    int bucketSize = 16;

    RandomGaussianCoordinates<double, CodeType> randomBox(nParticles, box);

    thrust::device_vector<CodeType> tree;
    thrust::device_vector<unsigned> counts;

    thrust::device_vector<CodeType> particleCodes(randomBox.mortonCodes().begin(),
                                                  randomBox.mortonCodes().end());

    auto tp0 = std::chrono::high_resolution_clock::now();

    computeOctreeGpu(thrust::raw_pointer_cast(particleCodes.data()),
                     thrust::raw_pointer_cast(particleCodes.data() + nParticles),
                     bucketSize,
                     tree, counts);

    auto tp1  = std::chrono::high_resolution_clock::now();

    double t0 = std::chrono::duration<double>(tp1 - tp0).count();
    std::cout << "build time from scratch " << t0 << " nNodes(tree): " << nNodes(tree)
              << " count: " << thrust::reduce(counts.begin(), counts.end(), 0) << std::endl;

    tp0  = std::chrono::high_resolution_clock::now();

    updateOctreeGpu(thrust::raw_pointer_cast(particleCodes.data()),
                    thrust::raw_pointer_cast(particleCodes.data() + nParticles),
                    bucketSize, tree, counts);

    tp1  = std::chrono::high_resolution_clock::now();

    double t1 = std::chrono::duration<double>(tp1 - tp0).count();

    std::cout << "build time with guess " << t1 << " nNodes(tree): " << nNodes(tree)
              << " count: " << thrust::reduce(counts.begin(), counts.end(), 0) << std::endl;
}
