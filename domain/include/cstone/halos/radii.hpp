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
 * @brief  Compute maximum interaction radii per octree cell as part of halo discovery
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/tree/octree.hpp"
#include "cstone/util/gsl-lite.hpp"

namespace cstone
{

/*! @brief Compute the halo radius of each node in the given octree
 *
 * This is the maximum distance beyond the node boundaries that a particle outside the
 * node could possibly interact with.
 *
 * TODO: Don't calculate the maximum smoothing length, calculate the maximum distance by
 *       which any of the particles plus radius protrude outside the node.
 *
 * @tparam Tin              float or double
 * @tparam Tout             float or double, usually float
 * @tparam KeyType          32- or 64-bit unsigned integer type for SFC codes
 * @tparam IndexType        integer type for local particle array indices, 32-bit for fewer than 2^32 local particles
 * @param[in]  tree         octree nodes given as SFC codes of length @a nNodes+1
 *                          This function does not rely on octree invariants, sortedness of the nodes
 *                          is the only requirement.
 * @param[in]  nNodes       number of nodes in tree
 * @param[in]  particleKeys sorted SFC particle keys
 * @param[in]  input        Radii per particle, i.e. the smoothing lengths in SPH, length = particleKeys.size()
 * @param[out] output       Radius per node, length = @a nNodes
 */
template<class KeyType, class Tin, class Tout>
void computeHaloRadii(const KeyType* tree, TreeNodeIndex nNodes, gsl::span<const KeyType> particleKeys,
                      const Tin* input, Tout* output)
{
    int firstNode = 0;
    int lastNode  = nNodes;
    if (!particleKeys.empty())
    {
        firstNode = std::upper_bound(tree, tree + nNodes, particleKeys.front()) - tree - 1;
        lastNode  = std::upper_bound(tree, tree + nNodes, particleKeys.back()) - tree;
    }

    #pragma omp parallel for schedule(static)
    for (TreeNodeIndex i = 0; i < firstNode; ++i)
        output[i] = 0;

    #pragma omp parallel for schedule(static)
    for (TreeNodeIndex i = lastNode; i < nNodes; ++i)
        output[i] = 0;

    #pragma omp parallel for
    for (TreeNodeIndex i = firstNode; i < lastNode; ++i)
    {
        KeyType nodeStart = tree[i];
        KeyType nodeEnd   = tree[i + 1];

        // find elements belonging to particles in node i
        LocalParticleIndex startIndex = findNodeAbove(particleKeys, nodeStart);
        LocalParticleIndex endIndex   = findNodeAbove(particleKeys, nodeEnd);

        Tin nodeMax = 0;
        for (LocalParticleIndex p = startIndex; p < endIndex; ++p)
        {
            nodeMax = std::max(nodeMax, input[p]);
        }

        // note factor of 2 due to SPH conventions
        output[i] = Tout(2 * nodeMax);
    }
}

} // namespace cstone
