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

#pragma once

#include <cuda_runtime.h>

#include "cstone/findneighbors.hpp"

template<class T, class I>
void findNeighborsCuda(const T* x, const T* y, const T* z, const T* h, int firstId, int lastId, int n,
                       cstone::Box<T> box, const I* codes, int* neighbors, int* neighborsCount, int ngmax,
                       cudaStream_t stream = cudaStreamDefault);


#define FIND_NEIGHBORS_CUDA(T, I) \
void findNeighborsCuda(const T* x, const T* y, const T* z, const T* h, int firstId, int lastId, int n, \
                       cstone::Box<T> box, const I* codes, int* neighbors, int* neighborsCount, int ngmax, \
                       cudaStream_t stream);

extern template FIND_NEIGHBORS_CUDA(float, uint32_t )
extern template FIND_NEIGHBORS_CUDA(float, uint64_t )
extern template FIND_NEIGHBORS_CUDA(double, uint32_t)
extern template FIND_NEIGHBORS_CUDA(double, uint64_t)

