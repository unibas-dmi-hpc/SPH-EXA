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

#include "cstone/util/tuple.hpp"
#include "cstone/util/util.hpp"

#include "sph/sph.cuh"
#include "sph/eos.hpp"
#include "sph/util/cuda_utils.cuh"

namespace sph
{
namespace cuda
{

template<class Tu, class Trho, class Tp, class Tc>
__global__ void cudaEOS_HydroStd(size_t firstParticle, size_t lastParticle, const Tu* u, const Trho* rho, Tp* p, Tc* c)
{
    unsigned i = firstParticle + blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= lastParticle) return;

    util::tie(p[i], c[i]) = idealGasEOS(u[i], rho[i]);
}

template<class Tu, class Trho, class Tp, class Tc>
void computeEOS_HydroStd(size_t firstParticle, size_t lastParticle, const Tu* u, const Trho* rho, Tp* p, Tc* c)
{
    int numThreads = 256;
    int numBlocks  = iceil(lastParticle - firstParticle, numThreads);
    cudaEOS_HydroStd<<<numBlocks, numThreads>>>(firstParticle, lastParticle, u, rho, p, c);
    CHECK_CUDA_ERR(cudaGetLastError());
}

template void computeEOS_HydroStd(size_t, size_t, const double*, const double*, double*, double*);
template void computeEOS_HydroStd(size_t, size_t, const double*, const float*, float*, float*);
template void computeEOS_HydroStd(size_t, size_t, const float*, const float*, float*, float*);

} // namespace cuda
} // namespace sph
