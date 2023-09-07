/*
 * MIT License
 *
 * Copyright (c) 2022 CSCS, ETH Zurich
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
 * @brief  Compute cell mass centers for use in focus tree
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "cstone/primitives/math.hpp"
#include "source_center.hpp"
#include "source_center_gpu.h"

namespace cstone
{

template<class Tc, class Tm, class Tf>
__global__ void computeLeafSourceCenterKernel(const Tc* x,
                                              const Tc* y,
                                              const Tc* z,
                                              const Tm* m,
                                              const TreeNodeIndex* leafToInternal,
                                              TreeNodeIndex numLeaves,
                                              const LocalIndex* layout,
                                              Vec4<Tf>* centers)
{
    TreeNodeIndex leafIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (leafIdx >= numLeaves) { return; }

    TreeNodeIndex nodeIdx = leafToInternal[leafIdx];
    centers[nodeIdx]      = massCenter<Tf>(x, y, z, m, layout[leafIdx], layout[leafIdx + 1]);
}

template<class Tc, class Tm, class Tf>
void computeLeafSourceCenterGpu(const Tc* x,
                                const Tc* y,
                                const Tc* z,
                                const Tm* m,
                                const TreeNodeIndex* leafToInternal,
                                TreeNodeIndex numLeaves,
                                const LocalIndex* layout,
                                Vec4<Tf>* centers)
{
    unsigned numThreads = 256;
    unsigned numBlocks  = iceil(numLeaves, numThreads);

    computeLeafSourceCenterKernel<<<numBlocks, numThreads>>>(x, y, z, m, leafToInternal, numLeaves, layout, centers);
}

template void computeLeafSourceCenterGpu(const double*,
                                         const double*,
                                         const double*,
                                         const double*,
                                         const TreeNodeIndex*,
                                         TreeNodeIndex,
                                         const LocalIndex*,
                                         Vec4<double>*);

template void computeLeafSourceCenterGpu(const double*,
                                         const double*,
                                         const double*,
                                         const float*,
                                         const TreeNodeIndex*,
                                         TreeNodeIndex,
                                         const LocalIndex*,
                                         Vec4<double>*);

template void computeLeafSourceCenterGpu(const double*,
                                         const double*,
                                         const double*,
                                         const float*,
                                         const TreeNodeIndex*,
                                         TreeNodeIndex,
                                         const LocalIndex*,
                                         Vec4<float>*);

template void computeLeafSourceCenterGpu(const float*,
                                         const float*,
                                         const float*,
                                         const float*,
                                         const TreeNodeIndex*,
                                         TreeNodeIndex,
                                         const LocalIndex*,
                                         Vec4<float>*);

template<class T>
__global__ void upsweepCentersKernel(TreeNodeIndex firstCell,
                                     TreeNodeIndex lastCell,
                                     const TreeNodeIndex* childOffsets,
                                     SourceCenterType<T>* centers)
{
    const int cellIdx = blockIdx.x * blockDim.x + threadIdx.x + firstCell;
    if (cellIdx >= lastCell) return;

    TreeNodeIndex firstChild = childOffsets[cellIdx];

    if (firstChild) { centers[cellIdx] = CombineSourceCenter<T>{}(cellIdx, firstChild, centers); }
}

template<class T>
void upsweepCentersGpu(int numLevels,
                       const TreeNodeIndex* levelRange,
                       const TreeNodeIndex* childOffsets,
                       SourceCenterType<T>* centers)
{
    constexpr int numThreads = 256;

    for (int level = numLevels - 1; level >= 0; level--)
    {
        int numCellsLevel = levelRange[level + 1] - levelRange[level];
        int numBlocks     = (numCellsLevel - 1) / numThreads + 1;
        if (numCellsLevel)
        {
            upsweepCentersKernel<<<numBlocks, numThreads>>>(levelRange[level], levelRange[level + 1], childOffsets,
                                                            centers);
        }
    }
}

template void upsweepCentersGpu(int, const TreeNodeIndex*, const TreeNodeIndex*, SourceCenterType<float>*);
template void upsweepCentersGpu(int, const TreeNodeIndex*, const TreeNodeIndex*, SourceCenterType<double>*);

template<class KeyType, class T>
__global__ void computeGeoCentersKernel(
    const KeyType* prefixes, TreeNodeIndex numNodes, Vec3<T>* centers, Vec3<T>* sizes, const Box<T> box)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNodes) { return; }

    KeyType prefix                  = prefixes[i];
    KeyType startKey                = decodePlaceholderBit(prefix);
    unsigned level                  = decodePrefixLength(prefix) / 3;
    auto nodeBox                    = sfcIBox(sfcKey(startKey), level);
    util::tie(centers[i], sizes[i]) = centerAndSize<KeyType>(nodeBox, box);
}

template<class KeyType, class T>
void computeGeoCentersGpu(
    const KeyType* prefixes, TreeNodeIndex numNodes, Vec3<T>* centers, Vec3<T>* sizes, const Box<T>& box)
{
    unsigned numThreads = 256;
    unsigned numBlocks  = iceil(numNodes, numThreads);
    computeGeoCentersKernel<<<numBlocks, numThreads>>>(prefixes, numNodes, centers, sizes, box);
}

template void computeGeoCentersGpu(const uint32_t*, TreeNodeIndex, Vec3<float>*, Vec3<float>*, const Box<float>&);
template void computeGeoCentersGpu(const uint32_t*, TreeNodeIndex, Vec3<double>*, Vec3<double>*, const Box<double>&);
template void computeGeoCentersGpu(const uint64_t*, TreeNodeIndex, Vec3<float>*, Vec3<float>*, const Box<float>&);
template void computeGeoCentersGpu(const uint64_t*, TreeNodeIndex, Vec3<double>*, Vec3<double>*, const Box<double>&);

} // namespace cstone