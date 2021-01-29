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

#include "../coord_samples/random.hpp"
#include "../../include/cstone/findneighbors.hpp"

__global__ void callClz(unsigned arg, int* ret)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid == 0)
    {
      *ret = countLeadingZeros(arg);
    }
}

int main()
{
    int* d_buffer;
    cudaMalloc((void**)&d_buffer, 4);
    //cudaMallocHost((void**)&buffer, 4);
    //cudaHostGetDevicePointer((void**)&d_buffer, buffer, 0);
    callClz<<<1,1>>>(2, d_buffer);

    int out;
    cudaMemcpy(&out, d_buffer, 4, cudaMemcpyDeviceToHost);
    std::cout << out << std::endl;

    cudaFree(d_buffer);

    using CodeType = unsigned;
    using T        = float;

    Box<T> box{0,1, true};
    int n = 2000000;

    RandomCoordinates<T, CodeType> coords(n, box);

    const T* x = coords.x().data();
    const T* y = coords.y().data();
    const T* z = coords.z().data();

    T* d_x;
    T* d_y;
    T* d_z;
    T* d_h;
    cudaMalloc((void **)&d_x, sizeof(T) * n);
    cudaMalloc((void **)&d_y, sizeof(T) * n);
    cudaMalloc((void **)&d_z, sizeof(T) * n);

    cudaMemcpy(d_x, x, sizeof(T) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, sizeof(T) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, z, sizeof(T) * n, cudaMemcpyHostToDevice);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
}
