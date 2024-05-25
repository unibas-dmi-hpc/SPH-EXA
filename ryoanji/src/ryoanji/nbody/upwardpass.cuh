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
 * @brief  Upsweep for multipole and source center computation
 *
 * @author Rio Yokota <rioyokota@gsic.titech.ac.jp>
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <chrono>

#include "cstone/cuda/gpu_config.cuh"

#include "kernel.hpp"

namespace ryoanji
{

struct UpsweepConfig
{
    static constexpr int numThreads = 256;
};

template<class Tc, class Tm, class Tf, class MType>
__global__ void computeLeafMultipoles(const Tc* x, const Tc* y, const Tc* z, const Tm* m,
                                      const TreeNodeIndex* leafToInternal, TreeNodeIndex numLeaves,
                                      const LocalIndex* layout, const Vec4<Tf>* centers, MType* multipoles)
{
    unsigned leafIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (leafIdx < numLeaves)
    {
        TreeNodeIndex i = leafToInternal[leafIdx];
        P2M(x, y, z, m, layout[leafIdx], layout[leafIdx + 1], centers[i], multipoles[i]);
    }
}

/*! @brief perform multipole upward sweep for one tree level
 *
 * launch config: one thread per cell of the current level
 *
 * @param[in]  firstCell        first cell to process
 * @param[in]  lastCell         last cell to process
 * @param[in]  childOffsets     cell index of first child of each node
 * @param[in]  centers          source expansion (mass) centers
 * @param[out] multipoles       output multipole of each cell
 */
template<class T, class MType>
__global__ void upsweepMultipoles(TreeNodeIndex firstCell, TreeNodeIndex lastCell, const TreeNodeIndex* childOffsets,
                                  const Vec4<T>* centers, MType* multipoles)
{
    const int cellIdx = blockIdx.x * blockDim.x + threadIdx.x + firstCell;
    if (cellIdx >= lastCell) return;

    TreeNodeIndex firstChild = childOffsets[cellIdx];

    // firstChild is zero if the cell is a leaf
    if (firstChild) { M2M(firstChild, firstChild + 8, centers[cellIdx], centers, multipoles, multipoles[cellIdx]); }
}

template<class Tc, class Tm, class Th, class Tf>
__global__ void computeLeafCenters(const Tc* x, const Tc* y, const Tc* z, const Tm* m, const Th* h,
                                   const TreeNodeIndex* leafToInternal, TreeNodeIndex numLeaves,
                                   const LocalIndex* layout, Vec4<Tf>* centers, Vec4<Tc>* cellXmin, Vec4<Tc>* cellXmax)
{
    unsigned leafIdx = blockIdx.x * blockDim.x + threadIdx.x;

    const Tc huge = Tc(1e10);
    Vec4<Tc> Xmin{+huge, +huge, +huge, +huge};
    Vec4<Tc> Xmax{-huge, -huge, -huge, -huge};
    Vec4<Tc> center;

    if (leafIdx < numLeaves)
    {
        TreeNodeIndex begin = layout[leafIdx];
        TreeNodeIndex end   = layout[leafIdx + 1];
        center              = setCenter(begin, end, x, y, z, m);
        for (int i = begin; i < end; i++)
        {
            Vec4<Tc> pos = {x[i], y[i], z[i], h[i]};
            Xmin         = min(Xmin, pos);
            Xmax         = max(Xmax, pos);
        }
        TreeNodeIndex cellIdx = leafToInternal[leafIdx];
        centers[cellIdx]      = center;
        cellXmin[cellIdx]     = Xmin;
        cellXmax[cellIdx]     = Xmax;
    }
}

/*! @brief perform source expansion center upward sweep for one tree level
 *
 * launch config: one thread per cell of the current level
 *
 * @param[in]  firstCell        first cell to process
 * @param[in]  lastCell         last cell to process
 * @param[in]  childOffsets     cell index of first child of each node
 * @param[out] centers          source expansion (mass) centers
 * @param[out] cellXmin         minimum coordinate of any body in the cell
 * @param[out] cellXmax         maximum coordinate of any body in the cell
 */
template<class T>
__global__ void upsweepCenters(TreeNodeIndex firstCell, TreeNodeIndex lastCell, const TreeNodeIndex* childOffsets,
                               Vec4<T>* centers, Vec4<T>* cellXmin, Vec4<T>* cellXmax)
{
    const int cellIdx = blockIdx.x * blockDim.x + threadIdx.x + firstCell;
    if (cellIdx >= lastCell) return;

    const T huge = T(1e10);
    Vec4<T> Xmin{+huge, +huge, +huge, +huge};
    Vec4<T> Xmax{-huge, -huge, -huge, -huge};
    Vec4<T> center;

    TreeNodeIndex firstChild = childOffsets[cellIdx];

    // firstChild is zero if the cell is a leaf
    if (firstChild)
    {
        TreeNodeIndex begin = firstChild;
        TreeNodeIndex end   = firstChild + 8;
        center              = setCenter(begin, end, centers);
        for (int i = begin; i < end; i++)
        {
            Xmin = min(Xmin, cellXmin[i]);
            Xmax = max(Xmax, cellXmax[i]);
        }
        centers[cellIdx]  = center;
        cellXmin[cellIdx] = Xmin;
        cellXmax[cellIdx] = Xmax;
    }
}

//! @brief calculate the squared MAC radius of each cell, store as the 4-th member sourceCenter
template<class T>
__global__ void setMAC(int numCells, T invTheta, Vec4<T>* sourceCenter, const Vec4<T>* cellXmin,
                       const Vec4<T>* cellXmax)
{
    int cellIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cellIdx >= numCells) { return; }

    Vec3<T> Xmin = makeVec3(cellXmin[cellIdx]);
    Vec3<T> Xmax = makeVec3(cellXmax[cellIdx]);
    T       Hmax = cellXmax[cellIdx][3];
    Vec3<T> Xi   = makeVec3(sourceCenter[cellIdx]);
    Vec3<T> X    = (Xmax + Xmin) * T(0.5);
    Vec3<T> R    = (Xmax - Xmin) * T(0.5);
    Vec3<T> dX   = X - Xi;
    T       s    = sqrt(norm2(dX));
    T       l    = T(2) * max(R);
    T       MAC  = max(l * invTheta + s, 2 * Hmax + s + l * 0.5);
    T       MAC2 = MAC * MAC;

    sourceCenter[cellIdx][3] = MAC2;
}

template<class MType>
__global__ void normalize(int numCells, MType* multipoles)
{
    using T     = typename MType::value_type;
    int cellIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cellIdx >= numCells) { return; }

    multipoles[cellIdx] = normalize(multipoles[cellIdx]);
}

template<class T, class MType>
void upsweep(int numSources, int numLeaves, int numLevels, T theta, const int2* levelRange, const T* x, const T* y,
             const T* z, const T* m, const T* h, const LocalIndex* layout, const TreeNodeIndex* childOffsets,
             const TreeNodeIndex* leafToInternal, Vec4<T>* centers, MType* Multipole)
{
    constexpr int numThreads = UpsweepConfig::numThreads;

    thrust::device_vector<Vec4<T>> d_cellXminmax(2 * numSources);
    Vec4<T>*                       cellXmin = rawPtr(d_cellXminmax);
    Vec4<T>*                       cellXmax = cellXmin + numSources;

    auto t0 = std::chrono::high_resolution_clock::now();

    computeLeafCenters<<<(numLeaves - 1) / numThreads + 1, numThreads>>>(x, y, z, m, h, leafToInternal, numLeaves,
                                                                         layout, centers, cellXmin, cellXmax);
    for (int level = numLevels - 1; level >= 1; level--)
    {
        int numCellsLevel = levelRange[level].y - levelRange[level].x;
        int numBlocks     = (numCellsLevel - 1) / numThreads + 1;
        upsweepCenters<<<numBlocks, numThreads>>>(levelRange[level].x, levelRange[level].y, childOffsets, centers,
                                                  cellXmin, cellXmax);
    }

    computeLeafMultipoles<<<(numLeaves - 1) / numThreads + 1, numThreads>>>(x, y, z, m, leafToInternal, numLeaves,
                                                                            layout, centers, Multipole);
    for (int level = numLevels - 1; level >= 1; level--)
    {
        int numCellsLevel = levelRange[level].y - levelRange[level].x;
        int numBlocks     = (numCellsLevel - 1) / numThreads + 1;
        upsweepMultipoles<<<numBlocks, numThreads>>>(levelRange[level].x, levelRange[level].y, childOffsets, centers,
                                                     Multipole);
    }

    kernelSuccess("upwardPass");

    int numBlocks = (numSources - 1) / numThreads + 1;
    setMAC<<<numBlocks, numThreads>>>(numSources, T(1.0) / theta, centers, cellXmin, cellXmax);
    normalize<<<numBlocks, numThreads>>>(numSources, Multipole);

    auto   t1 = std::chrono::high_resolution_clock::now();
    double dt = std::chrono::duration<double>(t1 - t0).count();

    fprintf(stdout, "Upward pass          : %.7f s\n", dt);
}

} // namespace ryoanji
