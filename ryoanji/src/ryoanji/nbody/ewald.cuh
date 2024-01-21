/*
 * MIT License
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *               2024 University of Basel
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
 * @brief Barnes-Hut breadth-first warp-aware tree traversal inspired by the original Bonsai implementation
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <cub/cub.cuh>

#include "ewald.hpp"
#include "traversal.cuh"

namespace ryoanji
{

__device__ float totalEwaldPotentialGlob = 0;

struct EwaldKernelConfig
{
    //! @brief number of threads per block for the Ewald kernel
    static constexpr int numThreads = 256;
};

template<class Tc, class Ta, class Tm, class Tmm>
__global__ void computeGravityEwaldKernel(LocalIndex first, LocalIndex last, const Tc* x, const Tc* y, const Tc* z,
                                          const Tm* m, float G, Ta* ugrav, Ta* ax, Ta* ay, Ta* az,
                                          EwaldParameters<Tc, Tmm> ewaldParams)
{
    LocalIndex i = first + blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= last) { return; }

    Vec3<Tc> target{x[i], y[i], z[i]};
    Vec4<Tc> potAcc{0, 0, 0, 0};

    potAcc += computeEwaldRealSpace(target, ewaldParams);
    potAcc += computeEwaldKSpace(target, ewaldParams);

    Ta Uewald = potAcc[0] * m[i];
    if (ugrav) { ugrav[i] += G * Uewald; } // potential per particle

    ax[i] += G * potAcc[1];
    ay[i] += G * potAcc[2];
    az[i] += G * potAcc[3];

    typedef cub::BlockReduce<Ta, EwaldKernelConfig::numThreads> BlockReduce;
    __shared__ typename BlockReduce::TempStorage                temp_storage;

    BlockReduce reduce(temp_storage);
    Ta          blockSum = reduce.Reduce(Uewald, cub::Sum());
    __syncthreads();

    if (threadIdx.x == 0) { atomicAdd(&totalEwaldPotentialGlob, blockSum); }
}

__global__ void resetEwaldPotential() { totalEwaldPotentialGlob = 0; }

//! GPU version of computeGravityEwald
template<class MType, class Tc, class Ta, class Tm, class Tu>
void computeGravityEwaldGpu(const cstone::Vec3<Tc>& rootCenter, const MType& Mroot, LocalIndex first, LocalIndex last,
                            const Tc* x, const Tc* y, const Tc* z, const Tm* m, const cstone::Box<Tc>& box, float G,
                            Ta* ugrav, Ta* ax, Ta* ay, Ta* az, Tu* ugravTot, int numReplicaShells, double lCut,
                            double hCut, double alpha_scale)
{
    if (box.minExtent() != box.maxExtent()) { throw std::runtime_error("Ewald gravity requires cubic bounding boxes"); }

    EwaldParameters<Tc, typename MType::value_type> ewaldParams =
        ewaldInitParameters(Mroot, rootCenter, numReplicaShells, box.lx(), lCut, hCut, alpha_scale);

    if (ewaldParams.numEwaldShells == 0) { return; }

    LocalIndex numTargets = last - first;
    unsigned   numThreads = EwaldKernelConfig::numThreads;
    unsigned   numBlocks  = (numTargets - 1) / numThreads + 1;

    resetEwaldPotential<<<1, 1>>>();
    computeGravityEwaldKernel<<<numBlocks, numThreads>>>(first, last, x, y, z, m, G, ugrav, ax, ay, az, ewaldParams);

    float totalPotential;
    checkGpuErrors(cudaMemcpyFromSymbol(&totalPotential, totalEwaldPotentialGlob, sizeof(float)));

    *ugravTot += 0.5 * G * totalPotential;
}

} // namespace ryoanji
