/*
 * MIT License
 *
 * Copyright (c) 2024 CSCS, ETH Zurich, University of Basel, University of Zurich
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
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUTh WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUTh NOTh LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENTh SHALL THE
 * AUTHORS OR COPYRIGHTh HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORTh OR OTHERWISE, ARISING FROM,
 * OUTh OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file calculates dB/dt with the induction equation, as well as dissipation and correction terms
 *
 * @author Lukas Schmidt
 */

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/findneighbors.hpp"
#include "cstone/traversal/find_neighbors.cuh"

#include "sph/sph_gpu.hpp"
#include "sph/particles_data.hpp"
#include "sph/magneto_ve/magneto_data.hpp"
#include "induction_dissipation.hpp"
#include "induction_dissipation_kern.hpp"

namespace sph::magneto::cuda
{
using cstone::GpuConfig;
using cstone::LocalIndex;
using cstone::TravConfig;
using cstone::TreeNodeIndex;

template<class Tc, class T, class Tm, class KeyType>
__global__ void inductionDissipationGPU(
    Tc K, unsigned ngmax, const cstone::Box<Tc> box, const LocalIndex* grpStart, const LocalIndex* grpEnd,
    LocalIndex numGroups, const cstone::OctreeNsView<Tc, KeyType> tree, const Tc mu_0, T Atmin, T Atmax, T ramp,
    const Tc* x, const Tc* y, const Tc* z, const T* vx, const T* vy, const T* vz, const Tc* Bx, const Tc* By,
    const Tc* Bz, const T* h, const T* c11, const T* c12, const T* c13, const T* c22, const T* c23, const T* c33,
    const T* wh, const T* xm, const T* kx, const T* gradh, const Tm* m, const T* c, const T* dvxdx, const T* dvxdy,
    const T* dvxdz, const T* dvydx, const T* dvydy, const T* dvydz, const T* dvzdx, const T* dvzdy, const T* dvzdz,
    T* psi, const T* curlB_x, const T* curlB_y, const T* curlB_z, const T* divB, Tc* dBx, Tc* dBy, Tc* dBz, Tc* du,
    T* ch_m1, T* d_psi, LocalIndex* nidx, TreeNodeIndex* globalPool)
{
    unsigned laneIdx     = threadIdx.x & (GpuConfig::warpSize - 1);
    unsigned targetIdx   = 0;
    unsigned warpIdxGrid = (blockDim.x * blockIdx.x + threadIdx.x) >> GpuConfig::warpSizeLog2;

    LocalIndex* neighborsWarp = nidx + ngmax * TravConfig::targetSize * warpIdxGrid;

    while (true)
    {
        // first thread in warp grabs next target
        if (laneIdx == 0) { targetIdx = atomicAdd(&cstone::targetCounterGlob, 1); }
        targetIdx = cstone::shflSync(targetIdx, 0);

        if (targetIdx >= numGroups) return;

        LocalIndex bodyBegin = grpStart[targetIdx];
        LocalIndex bodyEnd   = grpEnd[targetIdx];
        LocalIndex i         = bodyBegin + laneIdx;

        // Induction Equation
        dBx[i] = -Bx[i] * (dvydy[i] + dvzdz[i]) + By[i] * dvxdy[i] + Bz[i] * dvxdz[i];
        dBy[i] = -By[i] * (dvxdx[i] + dvzdz[i]) + Bx[i] * dvydx[i] + Bz[i] * dvydz[i];
        dBz[i] = -Bz[i] * (dvxdx[i] + dvydy[i]) + Bx[i] * dvzdx[i] + By[i] * dvzdy[i];

        auto ncTrue = traverseNeighbors(bodyBegin, bodyEnd, x, y, z, h, tree, box, neighborsWarp, ngmax, globalPool);

        if (i >= bodyEnd) continue;

        unsigned ncCapped = stl::min(ncTrue[0], ngmax);
        inductionDissipationJLoop<TravConfig::targetSize>(i, K, mu_0, Atmin, Atmax, ramp, box, neighborsWarp + laneIdx,
                                                          ncCapped, x, y, z, vx, vy, vz, Bx, By, Bz, h, c11, c12, c13,
                                                          c22, c23, c33, wh, xm, kx, gradh, m, psi, curlB_x, curlB_y,
                                                          curlB_z, divB, &dBx[i], &dBy[i], &dBz[i], &du[i]);

        // get psi time differential with the recipe of Wissing et al (2020)
        auto rho_i     = kx[i] * m[i] / xm[i];
        auto v_alfven2 = (Bx[i] * Bx[i] + By[i] * By[i] + Bz[i] * Bz[i]) / (mu_0 * rho_i);
        auto ch        = fclean * std::sqrt(c[i] * c[i] + v_alfven2);
        auto tau_Inv   = (sigma_c * ch) / h[i];
        d_psi[i]       = -ch_m1[i]*(ch * divB[i] + psi[i] * ((dvxdx[i] + dvydy[i] + dvzdz[i]) / 2 + tau_Inv) / ch);
        psi[i] *= ch / ch_m1[i];
        ch_m1[i] = ch;
    }
}

template<class HydroData, class MagnetoData>
void computeInductionDissipationGpu(const GroupView& grp, HydroData& d, MagnetoData& m,
                                    const cstone::Box<typename HydroData::RealType>& box)
{

    auto [traversalPool, nidxPool] = cstone::allocateNcStacks(d.devData.traversalStack, d.ngmax);
    cstone::resetTraversalCounters<<<1, 1>>>();

    inductionDissipationGPU<<<TravConfig::numBlocks(), TravConfig::numThreads>>>(
        d.K, d.ngmax, box, grp.groupStart, grp.groupEnd, grp.numGroups, d.treeView, m.mu_0, d.Atmin, d.Atmax, d.ramp,
        rawPtr(d.devData.x), rawPtr(d.devData.y), rawPtr(d.devData.z), rawPtr(d.devData.vx), rawPtr(d.devData.vy),
        rawPtr(d.devData.vz), rawPtr(m.devData.Bx), rawPtr(m.devData.By), rawPtr(m.devData.Bz), rawPtr(d.devData.h),
        rawPtr(d.devData.c11), rawPtr(d.devData.c12), rawPtr(d.devData.c13), rawPtr(d.devData.c22),
        rawPtr(d.devData.c23), rawPtr(d.devData.c33), rawPtr(d.devData.wh), rawPtr(d.devData.xm), rawPtr(d.devData.kx),
        rawPtr(d.devData.gradh), rawPtr(d.devData.m), rawPtr(d.devData.c), rawPtr(m.devData.dvxdx),
        rawPtr(m.devData.dvxdy), rawPtr(m.devData.dvxdz), rawPtr(m.devData.dvydx), rawPtr(m.devData.dvydy),
        rawPtr(m.devData.dvydz), rawPtr(m.devData.dvzdx), rawPtr(m.devData.dvzdy), rawPtr(m.devData.dvzdz),
        rawPtr(m.devData.psi), rawPtr(m.devData.curlB_x), rawPtr(m.devData.curlB_y), rawPtr(m.devData.curlB_z),
        rawPtr(m.devData.divB), rawPtr(m.devData.dBx), rawPtr(m.devData.dBy), rawPtr(m.devData.dBz),
        rawPtr(d.devData.du), rawPtr(m.devData.ch_m1), rawPtr(m.devData.d_psi), nidxPool, traversalPool);
}

template void computeInductionDissipationGpu(const GroupView& grp, sphexa::ParticlesData<cstone::GpuTag>& d,
                                             sphexa::magneto::MagnetoData<cstone::GpuTag>& m,
                                             const cstone::Box<SphTypes::CoordinateType>&);
} // namespace sph::magneto::cuda
