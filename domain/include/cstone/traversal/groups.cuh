/*
 * MIT License
 *
 * Copyright (c) 2023 CSCS, ETH Zurich, University of Basel, University of Zurich
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
 * @brief Particle target grouping
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "cstone/cuda/gpu_config.cuh"
#include "cstone/primitives/warpscan.cuh"
#include "cstone/tree/definitions.h"

namespace cstone
{

template<class T>
__global__ void groupTargets(LocalIndex first,
                             LocalIndex last,
                             const T* x,
                             const T* y,
                             const T* z,
                             const T* h,
                             unsigned groupSize,
                             LocalIndex* groups,
                             unsigned numGroups)
{
    LocalIndex tid = blockIdx.x * blockDim.x + threadIdx.x;

    LocalIndex groupIdx = tid / groupSize;
    if (groupIdx >= numGroups) { return; }

    if (tid == groupIdx * groupSize)
    {
        //LocalIndex i = first + tid;
        LocalIndex scan = first + groupIdx * groupSize;
        groups[groupIdx] = scan;
        if (groupIdx + 1 == numGroups)
        {
            groups[numGroups] = last;
        }
    }
}

} // namespace cstone