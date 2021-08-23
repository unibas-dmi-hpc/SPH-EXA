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
 * @brief Barnes-Hut tree walk to compute gravity forces on single particles
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/traversal/macs.hpp"
#include "cstone/traversal/upsweep.hpp"
#include "cstone/gravity/multipole.hpp"

namespace cstone
{

template<class KeyType, class T1, class T2, class T3>
void computeGravityGroup(TreeNodeIndex groupIdx,
                         const Octree<KeyType>& octree, const GravityMultipole<T1>* multipoles,
                         const LocalParticleIndex* layout,
                         const T2* x, const T2* y, const T2* z, const T3* m,
                         const Box<T2>& box, float theta, T2 eps2,
                         T2* ax, T2* ay, T2* az)
{
    auto treeLeaves     = octree.treeLeaves();
    KeyType groupKey    = treeLeaves[groupIdx];
    unsigned groupLevel = treeLevel(treeLeaves[groupIdx + 1] - groupKey);
    IBox groupBox       = sfcIBox(sfcKey(groupKey), groupLevel);

    float invThetaSq = 1.0 / (theta * theta);

    auto descendOrM2P = [groupIdx, multipoles, x, y, z, layout, invThetaSq, eps2, ax, ay, az, &octree, &groupBox,
                         &box](TreeNodeIndex idx)
    {
        // idx relative to root node
        KeyType nodeStart = octree.codeStart(idx);
        IBox sourceBox    = sfcIBox(sfcKey(nodeStart), octree.level(idx));

        bool violatesMac = !minDistanceMac<KeyType>(groupBox, sourceBox, box, invThetaSq);
        if (!violatesMac)
        {
            LocalParticleIndex firstTarget = layout[groupIdx];
            LocalParticleIndex lastTarget  = layout[groupIdx + 1];

            // apply multipole to all particles in group
            for (LocalParticleIndex t = firstTarget; t < lastTarget; ++t)
            {
                multipole2particle(x[t], y[t], z[t], multipoles[idx], eps2, ax + t, ay + t, az + t);
            }
        }

        return violatesMac;
    };

    auto leafP2P = [groupIdx, x, y, z, m, layout, eps2, ax, ay, az](TreeNodeIndex idx)
    {
        // idx relative to first leaf
        LocalParticleIndex firstTarget = layout[groupIdx];
        LocalParticleIndex lastTarget  = layout[groupIdx + 1];

        LocalParticleIndex firstSource = layout[idx];
        LocalParticleIndex numSources  = layout[idx + 1] - firstSource;

        if (groupIdx != idx)
        {
            // source node != target node
            for (LocalParticleIndex t = firstTarget; t < lastTarget; ++t)
            {
                particle2particle(x[t], y[t], z[t], x + firstSource, y + firstSource, z + firstSource, m + firstSource,
                                  numSources, eps2, ax + t, ay + t, az + t);
            }
        }
        else
        {
            assert(firstTarget == firstSource);
            // source node == target node -> source contains target, avoid self gravity
            for (LocalParticleIndex t = firstTarget; t < lastTarget; ++t)
            {
                // 2 splits: [firstSource:t] and [t+1:lastSource]
                particle2particle(x[t], y[t], z[t], x + firstSource, y + firstSource, z + firstSource, m + firstSource,
                                  t - firstSource, eps2, ax + t, ay + t, az + t);

                LocalParticleIndex tp1 = t + 1;
                particle2particle(x[t], y[t], z[t], x + tp1, y + tp1, z + tp1, m + tp1,
                                  numSources - tp1, eps2, ax + t, ay + t, az + t);
            }
        }
    };

    singleTraversal(octree, descendOrM2P, leafP2P);
}

template<class KeyType, class T1, class T2, class T3>
void computeGravity(const Octree<KeyType>& octree, const GravityMultipole<T1>* multipoles,
                    const LocalParticleIndex* layout, TreeNodeIndex firstLeafIndex, TreeNodeIndex lastLeafIndex,
                    const T2* x, const T2* y, const T2* z, const T3* m,
                    const Box<T2>& box, float theta, T2 eps2,
                    T2* ax, T2* ay, T2* az)
{
    #pragma omp parallel for
    for (TreeNodeIndex leafIdx = firstLeafIndex; leafIdx < lastLeafIndex; ++leafIdx)
    {
        computeGravityGroup(leafIdx, octree, multipoles, layout, x, y, z, m, box, theta, eps2, ax, ay, az);
    }
}

} // namespace cstone
