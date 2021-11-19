#pragma once

#include <chrono>

#include "kernel.h"

namespace
{

//! @brief computes the center of mass for the bodies in the specified range
__device__ __forceinline__ fvec4 setCenter(const int begin, const int end)
{
    fvec4 center;
    for (int i = begin; i < end; i++)
    {
        const fvec4 pos = tex1Dfetch(texBody, i);
        float weight    = pos[3];

        center[0] += weight * pos[0];
        center[1] += weight * pos[1];
        center[2] += weight * pos[2];
        center[3] += weight;
    }

    float invM = 1.0f / center[3];
    center[0] *= invM;
    center[1] *= invM;
    center[2] *= invM;

    return center;
}

//! @brief computes the center of mass for the bodies in the specified range
__host__ __device__ __forceinline__ fvec4 setCenter(const int begin, const int end, fvec4* posGlob)
{
    fvec4 center(0);
    for (int i = begin; i < end; i++)
    {
        fvec4 pos    = posGlob[i];
        float weight = pos[3];

        center[0] += weight * pos[0];
        center[1] += weight * pos[1];
        center[2] += weight * pos[2];
        center[3] += weight;
    }

    float invM = 1.0f / center[3];
    center[0] *= invM;
    center[1] *= invM;
    center[2] *= invM;

    return center;
}

/*! @brief perform multipole upward sweep for one tree level
 *
 * @param[in]  level            current level to process
 * @param[in]  levelRange       first and lasst node in @p cells of @p level
 * @param[in]  cells            the tree cells
 * @param[out] sourceCenter     the center of mass of each tree cell
 * @param[out] cellXmin         coordinate minimum of each cell
 * @param[out] cellXmax         coordinate maximum of each cell
 * @param[out] Multipole        output multipole of each cell
 */
__global__ __launch_bounds__(NTHREAD) void upwardPass(const int level, int2* levelRange, CellData* cells,
                                                      fvec4* sourceCenter, fvec3* cellXmin, fvec3* cellXmax,
                                                      fvec4* Multipole)
{
    const int cellIdx = blockIdx.x * blockDim.x + threadIdx.x + levelRange[level].x;
    if (cellIdx >= levelRange[level].y) return;
    const CellData cell = cells[cellIdx];
    const float huge    = 1e10f;
    fvec3 Xmin          = +huge;
    fvec3 Xmax          = -huge;
    fvec4 center;
    float M[4 * NVEC4];
    if (cell.isLeaf())
    {
        const int begin = cell.body();
        const int end   = begin + cell.nbody();
        center          = setCenter(begin, end);
        for (int i = begin; i < end; i++)
        {
            fvec3 pos = make_fvec3(fvec4(tex1Dfetch(texBody, i)));
            Xmin      = min(Xmin, pos);
            Xmax      = max(Xmax, pos);
        }
        P2M(begin, end, center, *(fvecP*)M);
    }
    else
    {
        const int begin = cell.child();
        const int end   = begin + cell.nchild();
        center          = setCenter(begin, end, sourceCenter);
        for (int i = begin; i < end; i++)
        {
            Xmin = min(Xmin, cellXmin[i]);
            Xmax = max(Xmax, cellXmax[i]);
        }
        M2M(begin, end, center, sourceCenter, Multipole, *(fvecP*)M);
    }
    sourceCenter[cellIdx] = center;
    cellXmin[cellIdx]     = Xmin;
    cellXmax[cellIdx]     = Xmax;
    for (int i = 0; i < NVEC4; i++)
        Multipole[NVEC4 * cellIdx + i] = fvec4(M[4 * i + 0], M[4 * i + 1], M[4 * i + 2], M[4 * i + 3]);
}

__global__ __launch_bounds__(NTHREAD) void setMAC(const int numCells, const float invTheta, fvec4* sourceCenter,
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
    const float s            = sqrt(norm(dX));
    const float l            = max(2.0f * max(R), 1.0e-6f);
    const float MAC          = l * invTheta + s;
    const float MAC2         = MAC * MAC;
    sourceCenter[cellIdx][3] = MAC2;
}

__global__ __launch_bounds__(NTHREAD) void normalize(const int numCells, fvec4* Multipole)
{
    const int cellIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cellIdx >= numCells) return;
    const float invM = 1.0 / Multipole[NVEC4 * cellIdx][0];
    Multipole[NVEC4 * cellIdx][1] *= invM;
    Multipole[NVEC4 * cellIdx][2] *= invM;
    Multipole[NVEC4 * cellIdx][3] *= invM;
    for (int i = 1; i < NVEC4; i++)
    {
        Multipole[NVEC4 * cellIdx + i] *= invM;
    }
}
} // namespace

class Pass
{
public:
    static void upward(const int numLeafs, const int numLevels, const float theta, cudaVec<int2>& levelRange,
                       cudaVec<fvec4>& bodyPos, cudaVec<CellData>& sourceCells, cudaVec<fvec4>& sourceCenter,
                       cudaVec<fvec4>& Multipole)
    {
        int numCells = sourceCells.size();
        int NBLOCK   = (numCells - 1) / NTHREAD + 1;
        bodyPos.bind(texBody);

        auto t0 = std::chrono::high_resolution_clock::now();

        cudaVec<fvec3> cellXmin(numCells);
        cudaVec<fvec3> cellXmax(numCells);

        levelRange.d2h();

        for (int level = numLevels; level >= 1; level--)
        {
            numCells = levelRange[level].y - levelRange[level].x;
            NBLOCK   = (numCells - 1) / NTHREAD + 1;
            upwardPass<<<NBLOCK, NTHREAD>>>(
                level, levelRange.d(), sourceCells.d(), sourceCenter.d(), cellXmin.d(), cellXmax.d(), Multipole.d());
            kernelSuccess("upwardPass");
        }

        numCells = sourceCells.size();
        NBLOCK   = (numCells - 1) / NTHREAD + 1;
        setMAC<<<NBLOCK, NTHREAD>>>(numCells, 1.0 / theta, sourceCenter.d(), cellXmin.d(), cellXmax.d());
        normalize<<<NBLOCK, NTHREAD>>>(numCells, Multipole.d());

        auto t1   = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(t1 - t0).count();

        fprintf(stdout, "Upward pass          : %.7f s\n", dt);

        bodyPos.unbind(texBody);
    }
};
