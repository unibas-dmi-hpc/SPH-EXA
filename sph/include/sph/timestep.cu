/*! @file
 * @brief GPU functions to manage target particle groups and block time-steps
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "cstone/primitives/math.hpp"
#include "sph/sph_gpu.hpp"

namespace sph
{

using cstone::LocalIndex;

template<class T>
__global__ void groupDivvKernel(float Krho, const LocalIndex* groups, size_t numSegments, const T* divv, float* groupDt)
{
    LocalIndex tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numSegments)
    {
        float localMax = -INFINITY;

        LocalIndex segStart = groups[tid];
        LocalIndex segEnd   = groups[tid + 1];

        for (LocalIndex i = segStart; i < segEnd; ++i)
        {
            localMax = max(localMax, divv[i]);
        }

        groupDt[tid] = min(groupDt[tid], Krho / abs(localMax));
    }
}

template<class T>
void groupDivvTimeStep(float Krho, const LocalIndex* groups, LocalIndex numGroups, const T* divv, float* groupDt)
{
    int numThreads = 256;
    int numBlocks  = cstone::iceil(numGroups, numThreads);

    groupDivvKernel<<<numBlocks, numThreads>>>(Krho, groups, numGroups, divv, groupDt);
}

template void groupDivvTimeStep(float, const LocalIndex*, LocalIndex, const float*, float*);
template void groupDivvTimeStep(float, const LocalIndex*, LocalIndex, const double*, float*);

template<class T>
__global__ void groupAccKernel(float etaAcc, const LocalIndex* groups, size_t numGroups, const T* ax, const T* ay,
                               const T* az, float* groupDt)
{
    LocalIndex tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numGroups)
    {
        float maxAcc = 0;

        LocalIndex segStart = groups[tid];
        LocalIndex segEnd   = groups[tid + 1];

        for (LocalIndex i = segStart; i < segEnd; ++i)
        {
            maxAcc = max(maxAcc, norm2(cstone::Vec3<T>{ax[i], ay[i], az[i]}));
        }

        groupDt[tid] = min(groupDt[tid], etaAcc / std::sqrt(maxAcc));
    }
}

template<class T>
void groupAccTimeStep(float etaAcc, const cstone::LocalIndex* groups, cstone::LocalIndex numGroups, const T* ax,
                      const T* ay, const T* az, float* groupDt)
{
    int numThreads = 256;
    int numBlocks  = cstone::iceil(numGroups, numThreads);

    groupAccKernel<<<numBlocks, numThreads>>>(etaAcc, groups, numGroups, ax, ay, az, groupDt);
}

template void groupAccTimeStep(float, const LocalIndex*, LocalIndex, const double*, const double*, const double*,
                               float*);
template void groupAccTimeStep(float, const LocalIndex*, LocalIndex, const float*, const float*, const float*, float*);

} // namespace sph
