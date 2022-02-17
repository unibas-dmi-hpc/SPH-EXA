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
 * @brief  A CPU tree-walk with given EXA-FMM multipoles to debug the GPU code
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <vector>

#include "cstone/gravity/multipole.hpp"

#include "ryoanji/kernel.hpp"
#include "ryoanji/traversal.cuh"
#include "ryoanji/types.h"

namespace ryoanji
{

fvec4 walkParticle(int targetIndex, std::size_t numBodies, float eps, const CellData* sources,
                   const fvec4* sourceCenter, const fvec4* Multipole, const fvec4* bodyPos)
{
    float EPS2 = eps * eps;
    fvec4 acc{0, 0, 0, 0};

    fvec3 targetX    = make_fvec3(bodyPos[targetIndex]);
    float targetMass = bodyPos[targetIndex][3];

    // initialize stack with level 1 nodes
    std::vector<int> stack{1, 2, 3, 4, 5, 6, 7, 8};
    stack.reserve(128);

    fvec3 gmin{1e10, 1e10, 1e10};
    fvec3 gmax{-1e10, -1e10, -1e10};

    int groupStart = targetIndex & ~63;
    int groupEnd   = std::min(groupStart + 64lu, numBodies);
    for (int i = groupStart; i < groupEnd; ++i)
    {
        gmin[0] = std::min(bodyPos[i][0], gmin[0]);
        gmin[1] = std::min(bodyPos[i][1], gmin[1]);
        gmin[2] = std::min(bodyPos[i][2], gmin[2]);
        gmax[0] = std::max(bodyPos[i][0], gmax[0]);
        gmax[1] = std::max(bodyPos[i][1], gmax[1]);
        gmax[2] = std::max(bodyPos[i][2], gmax[2]);
    }

    fvec3 targetCenter = (gmax + gmin) * 0.5f;
    fvec3 targetSize   = (gmax - gmin) * 0.5f;

    while (!stack.empty())
    {
        int ni = stack.back();
        stack.pop_back();

        fvec4 MAC = sourceCenter[ni];
        fvec3 srcCenter{MAC[0], MAC[1], MAC[2]};
        CellData srcData = sources[ni];
        bool isNode      = srcData.isNode();
        bool isClose     = applyMAC(srcCenter, MAC[3], srcData, targetCenter, targetSize);
        bool isSplit     = isNode && isClose;
        bool isLeaf      = !isNode;
        bool isDirect    = isLeaf && isClose;

        // MAC failed and not a leaf, push children on stack
        if (isSplit)
        {
            for (int i = 0; i < srcData.nchild(); ++i)
            {
                stack.push_back(srcData.child() + i);
            }
        }

        // MAC passed, can apply M2P
        if (!isClose)
        {
            const fvec4* M = &Multipole[NVEC4 * ni];
            acc            = M2P(acc, targetX, srcCenter, *(fvecP*)M, EPS2);
        }

        // MAC failed and is leaf, apply P2P
        if (isDirect)
        {
            for (int si = srcData.body(); si < srcData.body() + srcData.nbody(); ++si)
            {
                fvec4 srcBody = bodyPos[si];
                fvec3 srcX    = make_fvec3(srcBody);
                float mass    = srcBody[3];
                acc           = P2P(acc, targetX, srcX, mass, EPS2);
            }
        }
    }

    return acc;
}

} // namespace ryoanji
