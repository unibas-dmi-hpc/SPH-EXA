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

void demo()
{
    using I = unsigned;
    using T = double;

    std::vector<I> map{0,2,4,6,8,1,3,5,7,9};
    std::vector<T> values{0,1,2,3,4,5,6,7,8,9};

    DeviceGather<T, I> devGather;
    devGather.setReorderMap(map.data(), map.data() + map.size());
    devGather(values.data());

    std::vector<T> reference{0,2,4,6,8,1,3,5,7,9};

    if (reference == values)
        std::cout << "demo gather check: PASS\n";
    else
    {
        std::cout << "demo gather check: FAIL\n";

        std::cout << "expected: ";
        for (auto v : reference) std::cout << v << " ";
        std::cout << std::endl;

        std::cout << "actual: ";
        for (auto v : values) std::cout << v << " ";
        std::cout << std::endl;
    }
}

void setFromCodeDemo()
{
    using I = unsigned;
    using T = double;

    std::vector<I> codes{0, 50, 10, 60, 20, 70, 30, 80, 40, 90};

    DeviceGather<T, I> devGather;
    devGather.setMapFromCodes(codes.data(), codes.data() + codes.size());

    if (codes == std::vector<I>{0,10,20,30,40,50,60,70,80,90})
        std::cout << "demo code sort: PASS\n";
    else
    {
        std::cout << "demo code sort: FAIL\n";

        std::cout << "actual: ";
        for (auto v : codes) std::cout << v << " ";
        std::cout << std::endl;
    }

    std::vector<T> values{0,1,2,3,4,5,6,7,8,9};
    devGather(values.data());
    std::vector<T> reference{0,2,4,6,8,1,3,5,7,9};

    if (reference == values)
        std::cout << "demo code reorder check: PASS\n";
    else
    {
        std::cout << "demo code reorder check: FAIL\n";

        std::cout << "expected: ";
        for (auto v : reference) std::cout << v << " ";
        std::cout << std::endl;

        std::cout << "actual: ";
        for (auto v : values) std::cout << v << " ";
        std::cout << std::endl;
    }
}

template<class T, class I>
void reorderCheck(int nElements)
{
    std::vector<I> map = makeRandomPermutation<I>(nElements);
    std::vector<T> v(nElements);

    std::iota(begin(v), end(v), 0);

    std::vector<T> hv = v;

    auto tcpu0 = std::chrono::high_resolution_clock::now();
    cpuGather(map, hv);
    auto tcpu1 = std::chrono::high_resolution_clock::now();
    std::cout << "cpu gather Melements/s: " << T(nElements)/(1e6 * std::chrono::duration<double>(tcpu1 - tcpu0).count()) << std::endl;


    DeviceGather<T, I> devGather;
    devGather.setReorderMap(map.data(), map.data() + map.size());

    auto tgpu0 = std::chrono::high_resolution_clock::now();
    devGather(v.data());
    auto tgpu1 = std::chrono::high_resolution_clock::now();

    std::cout << "gpu gather Melements/s: " << T(nElements)/(1e6*std::chrono::duration<double>(tgpu1 - tgpu0).count()) << std::endl;

    bool pass = (v == hv);

    if (pass)
        std::cout << "gather check: PASS\n";
    else
        std::cout << "gather check: FAIL\n";
}

int main()
{
    using T = double;
    using I = unsigned;

    int nElements = 32000000;

    demo();
    setFromCodeDemo();
    reorderCheck<T, I>(nElements);
}
