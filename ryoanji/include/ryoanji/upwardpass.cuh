#pragma once

#include <chrono>

#include "kernel.hpp"

namespace ryoanji
{

struct UpsweepConfig
{
    static constexpr int numThreads = 256;
};

//! @brief computes the center of mass for the bodies in the specified range
__host__ __device__ __forceinline__ fvec4 setCenter(const int begin, const int end, const fvec4* posGlob)
{
    assert(begin <= end);

    fvec4 center{0, 0, 0, 0};
    for (int i = begin; i < end; i++)
    {
        fvec4 pos    = posGlob[i];
        float weight = pos[3];

        center[0] += weight * pos[0];
        center[1] += weight * pos[1];
        center[2] += weight * pos[2];
        center[3] += weight;
    }

    float invM = (center[3] != 0.0f) ? 1.0f / center[3] : 0.0f;
    center[0] *= invM;
    center[1] *= invM;
    center[2] *= invM;

    return center;
}

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
__global__ void upwardPass(const int firstCell, const int lastCell, CellData* cells, const fvec4* bodyPos,
                           fvec4* sourceCenter, fvec3* cellXmin, fvec3* cellXmax,
                           fvec4* Multipole)
{
    const int cellIdx = blockIdx.x * blockDim.x + threadIdx.x + firstCell;
    if (cellIdx >= lastCell) return;
    CellData& cell = cells[cellIdx];
    const float huge    = 1e10f;
    fvec3 Xmin{+huge, +huge, +huge};
    fvec3 Xmax{-huge, -huge, -huge};
    fvec4 center;
    float M[4 * NVEC4];

    for (int k = 0; k < 4 * NVEC4; ++k)
    {
        M[k] = 0;
    }

    if (cell.isLeaf())
    {
        const int begin = cell.body();
        const int end   = begin + cell.nbody();
        center          = setCenter(begin, end, bodyPos);
        for (int i = begin; i < end; i++)
        {
            fvec3 pos = make_fvec3(bodyPos[i]);
            Xmin      = min(Xmin, pos);
            Xmax      = max(Xmax, pos);
        }
        P2M(begin, end, center, bodyPos, *(fvecP*)M);
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

        M2M(begin, end, center, sourceCenter, Multipole, *(fvecP*)M);
    }
    sourceCenter[cellIdx] = center;
    cellXmin[cellIdx]     = Xmin;
    cellXmax[cellIdx]     = Xmax;
    for (int i = 0; i < NVEC4; i++)
        Multipole[NVEC4 * cellIdx + i] = fvec4{M[4 * i + 0], M[4 * i + 1], M[4 * i + 2], M[4 * i + 3]};
}

__global__ void setMAC(const int numCells, const float invTheta, fvec4* sourceCenter,
                       fvec3* cellXmin, fvec3* cellXmax)
{
    const int cellIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cellIdx >= numCells) return;
    const fvec3 Xmin         = cellXmin[cellIdx];
    const fvec3 Xmax         = cellXmax[cellIdx];
    const fvec3 Xi           = make_fvec3(sourceCenter[cellIdx]);
    const fvec3 X            = (Xmax + Xmin) * 0.5f;
    const fvec3 R            = (Xmax - Xmin) * 0.5f;
    const fvec3 dX           = X - Xi;
    const float s            = sqrt(norm2(dX));
    const float l            = max(2.0f * max(R), 1.0e-6f);
    const float MAC          = l * invTheta + s;
    const float MAC2         = MAC * MAC;
    sourceCenter[cellIdx][3] = MAC2;
}

__global__ void normalize(const int numCells, fvec4* Multipole)
{
    const int cellIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cellIdx >= numCells) return;

    float mass = Multipole[NVEC4 * cellIdx][0];
    float invM = (mass != 0.0f) ? 1.0 / mass : 0.0f;

    Multipole[NVEC4 * cellIdx][1] *= invM;
    Multipole[NVEC4 * cellIdx][2] *= invM;
    Multipole[NVEC4 * cellIdx][3] *= invM;

    for (int i = 1; i < NVEC4; i++)
    {
        Multipole[NVEC4 * cellIdx + i] *= invM;
    }
}

void upsweep(const int numSources, const int numLevels, const float theta, const int2* levelRange,
             const fvec4* bodyPos, CellData* sourceCells, fvec4* sourceCenter, fvec4* Multipole)
{
    constexpr int numThreads = UpsweepConfig::numThreads;

    thrust::device_vector<fvec3> d_cellXminmax(2 * numSources);
    fvec3* cellXmin = rawPtr(d_cellXminmax.data());
    fvec3* cellXmax = cellXmin + numSources;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int level = numLevels; level >= 1; level--)
    {
        int numCellsLevel = levelRange[level].y - levelRange[level].x;
        int numBlocks     = (numCellsLevel - 1) / numThreads + 1;
        upwardPass<<<numBlocks, numThreads>>>(levelRange[level].x,
                                              levelRange[level].y,
                                              sourceCells,
                                              bodyPos,
                                              sourceCenter,
                                              cellXmin,
                                              cellXmax,
                                              Multipole);
        kernelSuccess("upwardPass");
    }

    int numBlocks = (numSources - 1) / numThreads + 1;
    setMAC<<<numBlocks, numThreads>>>(numSources, 1.0 / theta, sourceCenter, cellXmin, cellXmax);
    normalize<<<numBlocks, numThreads>>>(numSources, Multipole);

    auto t1   = std::chrono::high_resolution_clock::now();
    double dt = std::chrono::duration<double>(t1 - t0).count();

    fprintf(stdout, "Upward pass          : %.7f s\n", dt);
}

} // namespace ryoanji
