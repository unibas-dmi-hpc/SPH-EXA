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
 * \brief  Tests a gather (reordering) operation on arrays
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "cstone/cuda/gather.cuh"

template<class I>
std::vector<I> makeRandomPermutation(std::size_t nElements)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<I> map(nElements);
    std::iota(begin(map), end(map), 0);
    std::shuffle(begin(map), end(map), gen);
    return map;
}

template<class T, class I>
void cpuGather(const std::vector<I>& map, std::vector<T>& values)
{
    std::vector<T> tmp(values.size());

    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < map.size(); ++i)
        tmp[i] = values[map[i]];

    using std::swap;
    swap(values, tmp);
}

void check()
{
    using I = unsigned;
    using T = double;

    std::vector<I> map{0,2,4,6,8,1,3,5,7,9};
    std::vector<T> values{0,1,2,3,4,5,6,7,8,9};

    DeviceGather<T, I> devGather;
    devGather.setReorderMap(map.data(), map.data() + map.size());
    //devGather(values.data());
    devGather.stage(values.data());
    devGather.gatherStaged();
    devGather.unstage(values.data());

    std::vector<T> reference{0,2,4,6,8,1,3,5,7,9};

    if (reference == values)
        std::cout << "gather check: PASS\n";
    else
    {
        std::cout << "gather check: fail\n";

        std::cout << "expected: ";
        for (auto v : reference) std::cout << v << " ";
        std::cout << std::endl;

        std::cout << "actual: ";
        for (auto v : values) std::cout << v << " ";
        std::cout << std::endl;
    }
}

template<class T, class I>
void multiArrayCheck(int nElements)
{
    std::vector<I> map = makeRandomPermutation<I>(nElements);
    std::vector<T> v1(nElements);
    std::vector<T> v2(nElements);
    std::vector<T> v3(nElements);
    std::vector<T> v4(nElements);

    std::iota(begin(v1), end(v1), 0);
    std::iota(begin(v2), end(v2), 0);
    std::iota(begin(v3), end(v3), 0);
    std::iota(begin(v4), end(v4), 0);

    std::vector<T> hv1 = v1;
    std::vector<T> hv2 = v2;
    std::vector<T> hv3 = v3;
    std::vector<T> hv4 = v4;

    std::array<T*, 4> arrays{v1.data(), v2.data(), v3.data(), v4.data()};

    DeviceGather<T, I> devGather;
    devGather.setReorderMap(map.data(), map.data() + map.size());

    auto tgpu0 = std::chrono::high_resolution_clock::now();
    devGather.reorderArrays(arrays.data(), arrays.size());
    auto tgpu1 = std::chrono::high_resolution_clock::now();

    std::cout << "gpu gather Melements/s: " << arrays.size() * T(nElements)/(1e6*std::chrono::duration<double>(tgpu1 - tgpu0).count()) << std::endl;

    auto tcpu0 = std::chrono::high_resolution_clock::now();
    cpuGather(map, hv1);
    cpuGather(map, hv2);
    cpuGather(map, hv3);
    cpuGather(map, hv4);
    auto tcpu1 = std::chrono::high_resolution_clock::now();

    bool pass = v1 == hv1 && v2 == hv2 && v3 == hv3 && v4 == hv4;

    std::cout << "cpu gather Melements/s: " << arrays.size() * T(nElements)/(1e6 * std::chrono::duration<double>(tcpu1 - tcpu0).count()) << std::endl;

    if (pass)
        std::cout << "multi array gather check: PASS";
    else
        std::cout << "multi array gather check: FAIL";
}

template<class T, class I>
void singleArrayCheck(int nElements)
{
    std::vector<I> map = makeRandomPermutation<I>(nElements);
    std::vector<T> values(nElements);

    auto tcpu0 = std::chrono::high_resolution_clock::now();
    cpuGather(map, values);
    auto tcpu1 = std::chrono::high_resolution_clock::now();

    std::cout << "cpu gather Melements/s: " << T(nElements)/(1e6 * std::chrono::duration<double>(tcpu1 - tcpu0).count()) << std::endl;

    DeviceGather<T, I> devGather;
    devGather.setReorderMap(map.data(), map.data() + map.size());

    devGather.stage(values.data());

    auto tgpu0 = std::chrono::high_resolution_clock::now();
    devGather.gatherStaged();
    auto tgpu1 = std::chrono::high_resolution_clock::now();

    devGather.unstage(values.data());

    std::cout << "gpu gather Melements/s: " << T(nElements)/(1e6*std::chrono::duration<double>(tgpu1 - tgpu0).count()) << std::endl;
}

int main()
{
    using T = double;
    using I = unsigned;

    int nElements = 3200000;

    check();

    //singleArrayCheck<T, I>(nElements);
    multiArrayCheck<T, I>(nElements);
}
