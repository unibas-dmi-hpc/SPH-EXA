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
 * @brief Density i-loop GPU driver
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "cstone/cuda/findneighbors.cuh"

#include "sph/sph.cuh"
#include "sph/kernels.hpp"
#include "sph/particles_data.hpp"
#include "sph/util/cuda_utils.cuh"
#include "sph/hydro_ve/momentum_energy_kern.hpp"

namespace sph
{
namespace cuda
{

struct GradPVEConfig
{
    //! @brief number of threads per block
    static constexpr int numThreads = 128;
};

__device__ float minDt_ve_device;

template<typename T, class KeyType>
__global__ void
momentumEnergyGpu(T sincIndex, T K, T Kcour, T Atmin, T Atmax, T ramp, int ngmax, const cstone::Box<T> box,
                  size_t first, size_t last, size_t numParticles, const KeyType* particleKeys, const T* x, const T* y,
                  const T* z, const T* vx, const T* vy, const T* vz, const T* h, const T* m, const T* prho, const T* c,
                  const T* c11, const T* c12, const T* c13, const T* c22, const T* c23, const T* c33, const T* wh,
                  const T* whd, const T* kx, const T* xm, const T* alpha, T* grad_P_x, T* grad_P_y, T* grad_P_z, T* du)
{
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned i   = tid + first;

    // need to hard-code ngmax stack allocation for now
    assert(ngmax <= NGMAX && "ngmax too big, please increase NGMAX to desired size");
    int neighbors[NGMAX];
    int neighborsCount;

    // starting from CUDA 11.3, dynamic stack allocation is available with the following command
    // int* neighbors = (int*)alloca(ngmax * sizeof(int));

    T dt_i = INFINITY;

    if (i < last)
    {
        cstone::findNeighbors(
            i, x, y, z, h, box, cstone::sfcKindPointer(particleKeys), neighbors, &neighborsCount, numParticles, ngmax);

        T maxvsignal;
        momentumAndEnergyJLoop(i,
                               sincIndex,
                               K,
                               box,
                               neighbors,
                               neighborsCount,
                               x,
                               y,
                               z,
                               vx,
                               vy,
                               vz,
                               h,
                               m,
                               prho,
                               c,
                               c11,
                               c12,
                               c13,
                               c22,
                               c23,
                               c33,
                               Atmin,
                               Atmax,
                               ramp,
                               wh,
                               whd,
                               kx,
                               xm,
                               alpha,
                               grad_P_x,
                               grad_P_y,
                               grad_P_z,
                               du,
                               &maxvsignal);

        dt_i = tsKCourant(maxvsignal, h[i], c[i], Kcour);
    }

    typedef cub::BlockReduce<T, GradPVEConfig::numThreads> BlockReduce;
    __shared__ typename BlockReduce::TempStorage           temp_storage;

    BlockReduce reduce(temp_storage);
    T           blockMin = reduce.Reduce(dt_i, cub::Min());
    __syncthreads();

    if (threadIdx.x == 0) { atomicMinFloat(&minDt_ve_device, blockMin); }
}

template<class Dataset>
void computeMomentumEnergy(size_t startIndex, size_t endIndex, size_t ngmax, Dataset& d,
                           const cstone::Box<typename Dataset::RealType>& box)
{
    size_t sizeWithHalos       = d.x.size();
    size_t numParticlesCompute = endIndex - startIndex;

    unsigned numThreads = 128;
    unsigned numBlocks  = (numParticlesCompute + numThreads - 1) / numThreads;

    float huge = 1e10;
    CHECK_CUDA_ERR(cudaMemcpyToSymbol(minDt_ve_device, &huge, sizeof(huge)));

    momentumEnergyGpu<<<numBlocks, numThreads>>>(d.sincIndex,
                                                 d.K,
                                                 d.Kcour,
                                                 d.Atmin,
                                                 d.Atmax,
                                                 d.ramp,
                                                 ngmax,
                                                 box,
                                                 startIndex,
                                                 endIndex,
                                                 sizeWithHalos,
                                                 rawPtr(d.devData.codes),
                                                 rawPtr(d.devData.x),
                                                 rawPtr(d.devData.y),
                                                 rawPtr(d.devData.z),
                                                 rawPtr(d.devData.vx),
                                                 rawPtr(d.devData.vy),
                                                 rawPtr(d.devData.vz),
                                                 rawPtr(d.devData.h),
                                                 rawPtr(d.devData.m),
                                                 rawPtr(d.devData.prho),
                                                 rawPtr(d.devData.c),
                                                 rawPtr(d.devData.c11),
                                                 rawPtr(d.devData.c12),
                                                 rawPtr(d.devData.c13),
                                                 rawPtr(d.devData.c22),
                                                 rawPtr(d.devData.c23),
                                                 rawPtr(d.devData.c33),
                                                 rawPtr(d.devData.wh),
                                                 rawPtr(d.devData.whd),
                                                 rawPtr(d.devData.kx),
                                                 rawPtr(d.devData.xm),
                                                 rawPtr(d.devData.alpha),
                                                 rawPtr(d.devData.ax),
                                                 rawPtr(d.devData.ay),
                                                 rawPtr(d.devData.az),
                                                 rawPtr(d.devData.du));
    CHECK_CUDA_ERR(cudaGetLastError());

    float minDt;
    CHECK_CUDA_ERR(cudaMemcpyFromSymbol(&minDt, minDt_ve_device, sizeof(minDt)));
    d.minDt_loc = minDt;
}

template void computeMomentumEnergy(size_t, size_t, size_t, sphexa::ParticlesData<double, unsigned, cstone::GpuTag>& d,
                                    const cstone::Box<double>&);
template void computeMomentumEnergy(size_t, size_t, size_t, sphexa::ParticlesData<double, uint64_t, cstone::GpuTag>& d,
                                    const cstone::Box<double>&);
template void computeMomentumEnergy(size_t, size_t, size_t, sphexa::ParticlesData<float, unsigned, cstone::GpuTag>& d,
                                    const cstone::Box<float>&);
template void computeMomentumEnergy(size_t, size_t, size_t, sphexa::ParticlesData<float, uint64_t, cstone::GpuTag>& d,
                                    const cstone::Box<float>&);

} // namespace cuda
} // namespace sph
