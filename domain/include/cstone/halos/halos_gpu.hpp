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
 * @brief  Implementation of halo discovery and halo exchange
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <vector>

#include "cstone/domain/layout.hpp"
#include "cstone/halos/exchange_halos_gpu.cuh"
#include "cstone/halos/halos.hpp"
#include "cstone/halos/radii.hpp"
#include "cstone/traversal/collisions.hpp"
#include "cstone/util/gsl-lite.hpp"
#include "cstone/util/index_ranges.hpp"

namespace cstone
{

template<class KeyType>
class HalosGpu
{
public:
    HalosGpu(int myRank)
        : myRank_(myRank)
    {
    }

    /*! @brief Discover which cells outside myRank's assignment are halos
     *
     * @param[in] focusedTree      Fully linked octree, focused on the assignment of the executing rank
     * @param[in] focusAssignment  Assignment of leaf tree cells to ranks
     * @param[in] particleKeys     Sorted view of locally owned particle keys, no halos
     * @param[in] box              Global coordinate bounding box
     * @param[in] h                smoothing lengths of locally owned particles
     */
    template<class T, class Th>
    void discover(const Octree<KeyType>& focusedTree,
                  gsl::span<const TreeIndexPair> focusAssignment,
                  gsl::span<const KeyType> particleKeys,
                  const Box<T> box,
                  const Th* h)
    {
        gsl::span<const KeyType> leaves = focusedTree.treeLeaves();
        TreeNodeIndex firstAssignedNode = focusAssignment[myRank_].start();
        TreeNodeIndex lastAssignedNode  = focusAssignment[myRank_].end();

        std::vector<float> haloRadii(nNodes(leaves));
        computeHaloRadii(leaves.data(), nNodes(leaves), particleKeys, h, haloRadii.data());

        reallocate(nNodes(leaves), haloFlags_);
        std::fill(begin(haloFlags_), end(haloFlags_), 0);
        findHalos(focusedTree, haloRadii.data(), box, firstAssignedNode, lastAssignedNode, haloFlags_.data());
    }

    /*! @brief Compute particle offsets of each tree node and determine halo send/receive indices
     *
     * @param[in]  leaves          (focus) tree leaves
     * @param[in]  counts          (focus) tree counts
     * @param[in]  assignment      assignment of @p leaves to ranks
     * @param[in]  particleKeys    sorted view of locally owned keys, without halos
     * @param[in]  peers           list of peer ranks
     * @param[out] layout          Particle offsets for each node in @p leaves w.r.t to the final particle buffers,
     *                             including the halos, length = counts.size() + 1. The last element contains
     *                             the total number of locally present particles, i.e. assigned + halos.
     *                             [layout[i]:layout[i+1]] indexes particles in the i-th leaf cell.
     *                             If the i-th cell is not a halo and not locally owned, its particles are not present
     *                             and the corresponding layout range has length zero.
     */
    void computeLayout(gsl::span<const KeyType> leaves,
                       gsl::span<const unsigned> counts,
                       gsl::span<const TreeIndexPair> assignment,
                       gsl::span<const KeyType> particleKeys,
                       gsl::span<const int> peers,
                       gsl::span<LocalIndex> layout)
    {
        computeNodeLayout(counts, haloFlags_, assignment[myRank_].start(), assignment[myRank_].end(), layout);
        auto newParticleStart = layout[assignment[myRank_].start()];
        auto newParticleEnd   = layout[assignment[myRank_].end()];

        outgoingHaloIndices_ =
            exchangeRequestKeys<KeyType>(leaves, haloFlags_, particleKeys, newParticleStart, assignment, peers);

        detail::checkHalos(myRank_, assignment, haloFlags_);
        detail::checkIndices(outgoingHaloIndices_, newParticleStart, newParticleEnd, layout.back());

        incomingHaloIndices_ = computeHaloReceiveList(layout, haloFlags_, assignment, peers);
    }

    /*! @brief repeat the halo exchange pattern from the previous sync operation for a different set of arrays
     *
     * @param[inout] arrays  std::vector<float or double> of size particleBufferSize_
     *
     * Arrays are not resized or reallocated. Function is const, but modifies mutable haloEpoch_ counter.
     */
    template<class... Arrays>
    void exchangeHalos(Arrays... arrays) const
    {
        haloexchange(haloEpoch_++, incomingHaloIndices_, outgoingHaloIndices_, arrays...);
    }

    gsl::span<int> haloFlags() { return haloFlags_; }

private:
    int myRank_;

    SendList incomingHaloIndices_;
    SendList outgoingHaloIndices_;

    std::vector<int> haloFlags_;

    /*! @brief Counter for halo exchange calls
     * Multiple client calls to domain::exchangeHalos() during a time-step
     * should get different MPI tags, because there is no global MPI_Barrier or MPI collective in between them.
     */
    mutable int haloEpoch_{0};
};

} // namespace cstone
