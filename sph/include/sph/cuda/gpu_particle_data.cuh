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
 * @brief Contains the object holding all particle data on the GPU
 */

#pragma once

#include <thrust/device_vector.h>

#include "cuda_utils.cuh"
#include "sph/pinned_allocator.h"
#include "sph/tables.hpp"
#include "sph/data_util.hpp"

namespace sphexa
{
namespace sph
{
namespace cuda
{

template<typename T, class KeyType>
class DeviceParticlesData
{
    size_t allocatedDeviceMemory = 0;
    size_t allocatedTaskSize     = 0;

public:
    // number of CUDA streams to use
    static constexpr int NST = 2;
    // max number of particles to process per launch in kernel with async transfers
    static constexpr int taskSize = 1000000;

    struct neighbors_stream
    {
        cudaStream_t stream;
        int*         d_neighborsCount;
    };

    struct neighbors_stream d_stream[NST];

    T *d_x, *d_y, *d_z;
    T *d_vx, *d_vy, *d_vz;
    T *d_h;
    T *d_m;
    T *d_wh, *d_whd;
    T *d_rho0;
    T *d_wrho0;
    T *d_rho;
    T *d_kx;
    T *d_whomega;
    T *d_p;
    T *d_c;
    T *d_c11, *d_c12, *d_c13, *d_c22, *d_c23, *d_c33;
    T *d_divv, *d_curlv;
    T *d_alpha;
    T *d_grad_P_x, *d_grad_P_y, *d_grad_P_z;
    T *d_du;
    T *d_maxvsignal;

    KeyType* d_codes;

    DeviceParticlesData()
    {
        size_t                        size_lt_T = lt::size * sizeof(T);
        const std::array<T, lt::size> wh        = lt::createWharmonicLookupTable<T, lt::size>();
        const std::array<T, lt::size> whd       = lt::createWharmonicDerivativeLookupTable<T, lt::size>();

        CHECK_CUDA_ERR(utils::cudaMalloc(size_lt_T, d_wh, d_whd));
        CHECK_CUDA_ERR(cudaMemcpy(d_wh, wh.data(), size_lt_T, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(d_whd, whd.data(), size_lt_T, cudaMemcpyHostToDevice));

        for (int i = 0; i < NST; ++i)
        {
            CHECK_CUDA_ERR(cudaStreamCreate(&d_stream[i].stream));
        }
        resize_streams(taskSize);
    }

    ~DeviceParticlesData()
    {
        CHECK_CUDA_ERR(utils::cudaFree(d_x, d_y, d_z));
        CHECK_CUDA_ERR(utils::cudaFree(d_vx, d_vy, d_vz));
        CHECK_CUDA_ERR(utils::cudaFree(d_h));
        CHECK_CUDA_ERR(utils::cudaFree(d_m));
        CHECK_CUDA_ERR(utils::cudaFree(d_wh, d_whd));
        CHECK_CUDA_ERR(utils::cudaFree(d_rho0));
        CHECK_CUDA_ERR(utils::cudaFree(d_wrho0));
        CHECK_CUDA_ERR(utils::cudaFree(d_rho));
        CHECK_CUDA_ERR(utils::cudaFree(d_kx));
        CHECK_CUDA_ERR(utils::cudaFree(d_whomega));
        CHECK_CUDA_ERR(utils::cudaFree(d_p));
        CHECK_CUDA_ERR(utils::cudaFree(d_c));
        CHECK_CUDA_ERR(utils::cudaFree(d_c11, d_c12, d_c13, d_c22, d_c23, d_c33));
        CHECK_CUDA_ERR(utils::cudaFree(d_divv, d_curlv));
        CHECK_CUDA_ERR(utils::cudaFree(d_alpha));
        CHECK_CUDA_ERR(utils::cudaFree(d_grad_P_x,d_grad_P_y,d_grad_P_z));
        CHECK_CUDA_ERR(utils::cudaFree(d_du));
        CHECK_CUDA_ERR(utils::cudaFree(d_maxvsignal));

        CHECK_CUDA_ERR(utils::cudaFree(d_codes));

        for (int i = 0; i < NST; ++i)
        {
            CHECK_CUDA_ERR(cudaStreamDestroy(d_stream[i].stream));
            CHECK_CUDA_ERR(utils::cudaFree(d_stream[i].d_neighborsCount));
        }
    }

    void resize(size_t size)
    {
        if (size > allocatedDeviceMemory)
        {
            // TODO: Investigate benefits of low-level reallocate
            if (allocatedDeviceMemory)
            {
                CHECK_CUDA_ERR(utils::cudaFree(d_x, d_y, d_z));
                CHECK_CUDA_ERR(utils::cudaFree(d_vx, d_vy, d_vz));
                CHECK_CUDA_ERR(utils::cudaFree(d_h));
                CHECK_CUDA_ERR(utils::cudaFree(d_m));
                CHECK_CUDA_ERR(utils::cudaFree(d_rho0));
                CHECK_CUDA_ERR(utils::cudaFree(d_wrho0));
                CHECK_CUDA_ERR(utils::cudaFree(d_rho));
                CHECK_CUDA_ERR(utils::cudaFree(d_kx));
                CHECK_CUDA_ERR(utils::cudaFree(d_whomega));
                CHECK_CUDA_ERR(utils::cudaFree(d_p));
                CHECK_CUDA_ERR(utils::cudaFree(d_c));
                CHECK_CUDA_ERR(utils::cudaFree(d_c11, d_c12, d_c13, d_c22, d_c23, d_c33));
                CHECK_CUDA_ERR(utils::cudaFree(d_divv, d_curlv));
                CHECK_CUDA_ERR(utils::cudaFree(d_alpha));
                CHECK_CUDA_ERR(utils::cudaFree(d_grad_P_x,d_grad_P_y,d_grad_P_z));
                CHECK_CUDA_ERR(utils::cudaFree(d_du));
                CHECK_CUDA_ERR(utils::cudaFree(d_maxvsignal));

                CHECK_CUDA_ERR(utils::cudaFree(d_codes));
            }

            size = size_t(double(size) * 1.01); // allocate 1% extra to avoid reallocation on small size increase

            size_t size_np_T       = size * sizeof(T);
            CHECK_CUDA_ERR(utils::cudaMalloc(size_np_T, d_x, d_y, d_z));
            CHECK_CUDA_ERR(utils::cudaMalloc(size_np_T, d_vx, d_vy, d_vz));
            CHECK_CUDA_ERR(utils::cudaMalloc(size_np_T, d_h));
            CHECK_CUDA_ERR(utils::cudaMalloc(size_np_T, d_m));
            CHECK_CUDA_ERR(utils::cudaMalloc(size_np_T, d_rho0));
            CHECK_CUDA_ERR(utils::cudaMalloc(size_np_T, d_wrho0));
            CHECK_CUDA_ERR(utils::cudaMalloc(size_np_T, d_rho));
            CHECK_CUDA_ERR(utils::cudaMalloc(size_np_T, d_kx));
            CHECK_CUDA_ERR(utils::cudaMalloc(size_np_T, d_whomega));
            CHECK_CUDA_ERR(utils::cudaMalloc(size_np_T, d_p));
            CHECK_CUDA_ERR(utils::cudaMalloc(size_np_T, d_c));
            CHECK_CUDA_ERR(utils::cudaMalloc(size_np_T, d_c11, d_c12, d_c13, d_c22, d_c23, d_c33));
            CHECK_CUDA_ERR(utils::cudaMalloc(size_np_T, d_divv, d_curlv));
            CHECK_CUDA_ERR(utils::cudaMalloc(size_np_T, d_alpha));
            CHECK_CUDA_ERR(utils::cudaMalloc(size_np_T, d_grad_P_x,d_grad_P_y,d_grad_P_z));
            CHECK_CUDA_ERR(utils::cudaMalloc(size_np_T, d_du));
            CHECK_CUDA_ERR(utils::cudaMalloc(size_np_T, d_maxvsignal));

            size_t size_np_KeyType = size * sizeof(KeyType);
            CHECK_CUDA_ERR(utils::cudaMalloc(size_np_KeyType, d_codes));

            allocatedDeviceMemory = size;
        }
    }

    void resize_streams(size_t taskSize)
    {
        if (taskSize > allocatedTaskSize)
        {
            if (allocatedTaskSize)
            {
                // printf("[D] increased stream size from %ld to %ld\n", allocatedTaskSize, taskSize);
                for (int i = 0; i < NST; ++i)
                {
                    CHECK_CUDA_ERR(utils::cudaFree(d_stream[i].d_neighborsCount));
                }
            }

            taskSize =
                size_t(double(taskSize) * 1.01); // allocate 1% extra to avoid reallocation on small size increase

            for (int i = 0; i < NST; ++i)
            {
                CHECK_CUDA_ERR(utils::cudaMalloc(taskSize * sizeof(int), d_stream[i].d_neighborsCount));
            }

            allocatedTaskSize = taskSize;
        }
    }
};
} // namespace cuda
} // namespace sph
} // namespace sphexa
