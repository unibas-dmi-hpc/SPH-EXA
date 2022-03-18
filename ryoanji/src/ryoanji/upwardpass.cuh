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
 */

#pragma once

#include <chrono>

#include "kernel.hpp"

namespace ryoanji
{

struct UpsweepConfig
{
    static constexpr int numThreads = 256;
};

/*! @brief perform multipole upward sweep for one tree level
 *
 * launch config: one thread per cell of the current level
 *
 * @param[in]  level            current level to process
 * @param[in]  levelRange       first and last node in @p cells of @p level
 * @param[in]  cells            the tree cells
 * @param[in]  bodyPos          SFC sorted bodies as referenced by @p cells
 * @param[out] sourceCenter     the center of mass of each tree cell
 * @param[out] cellXmin         coordinate minimum of each cell
 * @param[out] cellXmax         coordinate maximum of each cell
 * @param[out] Multipole        output multipole of each cell
 */
template<class T, class MType>
__global__ void upwardPass(const int firstCell, const int lastCell, CellData* cells, const T* x, const T* y, const T* z,
                           const T* m, Vec4<T>* sourceCenter, Vec3<T>* cellXmin, Vec3<T>* cellXmax, MType* Multipole)
{
    const int cellIdx = blockIdx.x * blockDim.x + threadIdx.x + firstCell;
    if (cellIdx >= lastCell) return;
    CellData& cell = cells[cellIdx];
    const T   huge = T(1e10);
    Vec3<T>   Xmin{+huge, +huge, +huge};
    Vec3<T>   Xmax{-huge, -huge, -huge};
    Vec4<T>   center;
    MType     M;
    M = 0;

    if (cell.isLeaf())
    {
        const int begin = cell.body();
        const int end   = begin + cell.nbody();
        center          = setCenter(begin, end, x, y, z, m);
        for (int i = begin; i < end; i++)
        {
            Vec3<T> pos = {x[i], y[i], z[i]};
            Xmin        = min(Xmin, pos);
            Xmax        = max(Xmax, pos);
        }
        P2M(begin, end, center, x, y, z, m, M);
    }
    else
    {
        const int begin = cell.child();
        const int end   = begin + cell.nchild();
        center          = setCenter(begin, end, sourceCenter);

        unsigned numBodiesChildren = 0;
        for (int i = begin; i < end; i++)
        {
            Xmin = min(Xmin, cellXmin[i]);
            Xmax = max(Xmax, cellXmax[i]);
            numBodiesChildren += cells[i].nbody();
        }

        cell.setBody(cells[begin].body());
        cell.setNBody(numBodiesChildren);

        M2M(begin, end, center, sourceCenter, Multipole, M);
    }
    sourceCenter[cellIdx] = center;
    cellXmin[cellIdx]     = Xmin;
    cellXmax[cellIdx]     = Xmax;
    Multipole[cellIdx]    = M;
}

template<class T>
__global__ void setMAC(int numCells, T invTheta, Vec4<T>* sourceCenter, const Vec3<T>* cellXmin,
                       const Vec3<T>* cellXmax)
{
    int cellIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cellIdx >= numCells) { return; }

    Vec3<T> Xmin = cellXmin[cellIdx];
    Vec3<T> Xmax = cellXmax[cellIdx];
    Vec3<T> Xi   = makeVec3(sourceCenter[cellIdx]);
    Vec3<T> X    = (Xmax + Xmin) * T(0.5);
    Vec3<T> R    = (Xmax - Xmin) * T(0.5f);
    Vec3<T> dX   = X - Xi;
    T       s    = sqrt(norm2(dX));
    T       l    = max(T(2.0) * max(R), T(1.0e-6));
    T       MAC  = l * invTheta + s;
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
void upsweep(int numSources, int numLevels, T theta, const int2* levelRange, const T* x, const T* y, const T* z,
             const T* m, CellData* sourceCells, Vec4<T>* sourceCenter, MType* Multipole)
{
    constexpr int numThreads = UpsweepConfig::numThreads;

    thrust::device_vector<Vec3<T>> d_cellXminmax(2 * numSources);
    Vec3<T>*                       cellXmin = rawPtr(d_cellXminmax.data());
    Vec3<T>*                       cellXmax = cellXmin + numSources;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int level = numLevels; level >= 1; level--)
    {
        int numCellsLevel = levelRange[level].y - levelRange[level].x;
        int numBlocks     = (numCellsLevel - 1) / numThreads + 1;
        upwardPass<<<numBlocks, numThreads>>>(levelRange[level].x,
                                              levelRange[level].y,
                                              sourceCells,
                                              x,
                                              y,
                                              z,
                                              m,
                                              sourceCenter,
                                              cellXmin,
                                              cellXmax,
                                              Multipole);
        kernelSuccess("upwardPass");
    }

    int numBlocks = (numSources - 1) / numThreads + 1;
    setMAC<<<numBlocks, numThreads>>>(numSources, T(1.0) / theta, sourceCenter, cellXmin, cellXmax);
    normalize<<<numBlocks, numThreads>>>(numSources, Multipole);

    auto   t1 = std::chrono::high_resolution_clock::now();
    double dt = std::chrono::duration<double>(t1 - t0).count();

    fprintf(stdout, "Upward pass          : %.7f s\n", dt);
}

} // namespace ryoanji
