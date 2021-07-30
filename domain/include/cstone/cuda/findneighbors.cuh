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
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <cuda_runtime.h>

#include "cstone/findneighbors.hpp"

/*! @brief find neighbors on the GPU
 *
 * @tparam     T               float or double
 * @tparam     Integer         32- or 64-bit unsigned integer
 * @param[in]  x               device x-coords, size @p n, order consistent with @p coords
 * @param[in]  y               device y-coords, size @p n, order consistent with @p coords
 * @param[in]  z               device z-coords, size @p n, order consistent with @p coords
 * @param[in]  h               device h-radii,  size @p n, order consistent with @p coords
 * @param[in]  firstId         first particle index in [0:n] for which to compute neighbors
 * @param[in]  lastId          last particle indes in [0:n] for which to compute neighbors
 * @param[in]  n               number of coordinates in x,y,z,h
 * @param[in]  box             coordinate bounding box used to calculate codes
 * @param[in]  particleKeys    device particle SFC keys, sorted, size @p n
 * @param[out] neighbors       device neighbor indices found per particle
 * @param[out] neighborsCount  device number of neighbors found per particles
 * @param[in]  ngmax           maximum number of neighbors to store in @p neighbors
 * @param[in]  stream          execute on cuda stream @p stream
 *
 * Preconditions:
 *      - codes[i] = sfc3D(x[i], y[i], z[i], box);
 *      - codes is sorted
 *
 * Postconditions:
 *      - If id is the index of the particle (x[id], y[id], z[id), and if id is in [firstId:lastId], then
 *        the neighbors of id are stored in
 *        neighbors[(id-firstId)*ngmax, (id-firstId)*ngmax + neighborsCount[id-firstId]]
 */
template<class T, class Integer>
void findNeighborsMortonGpu(const T* x, const T* y, const T* z, const T* h, int firstId, int lastId, int n,
                            cstone::Box<T> box, const Integer* particleKeys,
                            int* neighbors, int* neighborsCount, int ngmax,
                            cudaStream_t stream = cudaStreamDefault);


#define FIND_NEIGHBORS_MORTON_GPU(T, Integer) \
void findNeighborsMortonGpu(const T* x, const T* y, const T* z, const T* h, int firstId, int lastId, int n, \
                            cstone::Box<T> box, const Integer* particleKeys, int* neighbors, int* neighborsCount, \
                            int ngmax, cudaStream_t stream);

template<class T, class Integer>
void findNeighborsHilbertGpu(const T* x, const T* y, const T* z, const T* h, int firstId, int lastId, int n,
                             cstone::Box<T> box, const Integer* particleKeys,
                             int* neighbors, int* neighborsCount, int ngmax,
                             cudaStream_t stream = cudaStreamDefault);


#define FIND_NEIGHBORS_HILBERT_GPU(T, Integer) \
void findNeighborsHilbertGpu(const T* x, const T* y, const T* z, const T* h, int firstId, int lastId, int n, \
cstone::Box<T> box, const Integer* particleKeys, int* neighbors, int* neighborsCount, \
int ngmax, cudaStream_t stream);

extern template FIND_NEIGHBORS_MORTON_GPU(float,  uint32_t)
extern template FIND_NEIGHBORS_MORTON_GPU(float,  uint64_t)
extern template FIND_NEIGHBORS_MORTON_GPU(double, uint32_t)
extern template FIND_NEIGHBORS_MORTON_GPU(double, uint64_t)

extern template FIND_NEIGHBORS_HILBERT_GPU(float,  uint32_t)
extern template FIND_NEIGHBORS_HILBERT_GPU(float,  uint64_t)
extern template FIND_NEIGHBORS_HILBERT_GPU(double, uint32_t)
extern template FIND_NEIGHBORS_HILBERT_GPU(double, uint64_t)
