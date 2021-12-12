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
 * @brief  CPU driver for halo discovery using traversal of an internal binary radix tree
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <vector>

#include "cstone/domain/layout.hpp"
#include "cstone/halos/exchange_halos.hpp"
#include "cstone/traversal/collisions.hpp"
#include "cstone/util/gsl-lite.hpp"
#include "cstone/util/index_ranges.hpp"

namespace cstone
{

template<class KeyType>
class Halos
{
public:
    Halos() = default;

    template<class T, class Th>
    void discover(const Octree<KeyType>& focusedTree,
                  gsl::span<const unsigned> focusLeafCounts,
                  gsl::span<const TreeIndexPair> focusAssignment,
                  gsl::span<const KeyType> particleKeys,
                  int myRank,
                  gsl::span<const int> peers,
                  const Box<T> box,
                  const Th* h,
                  const LocalParticleIndex* sfcOrder,
                  LocalParticleIndex compactOffset,
                  gsl::span<LocalParticleIndex> layout)
    {
        gsl::span<const KeyType> leaves = focusedTree.treeLeaves();
        TreeNodeIndex firstAssignedNode = focusAssignment[myRank].start();
        TreeNodeIndex lastAssignedNode = focusAssignment[myRank].end();

        std::vector<float> haloRadii(nNodes(leaves));
        computeHaloRadii(leaves.data(),
                         nNodes(leaves),
                         particleKeys,
                         sfcOrder + compactOffset,
                         h,
                         haloRadii.data());

        std::vector<int> haloFlags(nNodes(leaves), 0);
        findHalos(focusedTree,
                  haloRadii.data(),
                  box,
                  firstAssignedNode,
                  lastAssignedNode,
                  haloFlags.data());

        layout = computeNodeLayout(focusLeafCounts, haloFlags, firstAssignedNode, lastAssignedNode);

        auto newParticleStart = layout[firstAssignedNode];
        auto newParticleEnd   = newParticleStart + newNParticlesAssigned;

        outgoingHaloIndices_ = exchangeRequestKeys<KeyType>(leaves, haloFlags, particleKeys,
                                                            newParticleStart, focusAssignment, peers);
        checkIndices(outgoingHaloIndices_, newParticleStart, newParticleEnd);

        incomingHaloIndices_ = computeHaloReceiveList(layout, haloFlags, focusAssignment, peers);
    }

    /*! @brief repeat the halo exchange pattern from the previous sync operation for a different set of arrays
     *
     * @param[inout] arrays  std::vector<float or double> of size localNParticles_
     *
     * Arrays are not resized or reallocated. This is used e.g. for densities.
     * Note: function is const, but modiefies mutable haloEpoch_ counter.
     */
    template<class... Arrays>
    void exchangeHalos(Arrays&... arrays) const
    {
        if (!sizesAllEqualTo(localNParticles_, arrays...))
        {
            throw std::runtime_error("halo exchange array sizes inconsistent with previous sync operation\n");
        }

        haloexchange(haloEpoch_++, incomingHaloIndices_, outgoingHaloIndices_, arrays.data()...);
    }

    void resetEpochs() { haloEpoch_ = 0; }

private:
    //! @brief check that only owned particles in [particleStart_:particleEnd_] are sent out as halos
    void checkIndices(const SendList& sendList, LocalParticleIndex start, LocalParticleIndex end)
    {
        for (const auto& manifest : sendList)
        {
            for (size_t ri = 0; ri < manifest.nRanges(); ++ri)
            {
                assert(!overlapTwoRanges(LocalParticleIndex{0}, start, manifest.rangeStart(ri), manifest.rangeEnd(ri)));
                assert(!overlapTwoRanges(end, localNParticles_, manifest.rangeStart(ri), manifest.rangeEnd(ri)));
            }
        }
    }

    SendList incomingHaloIndices_;
    SendList outgoingHaloIndices_;

    /*! @brief counter for halo exchange calls between sync() calls
     *
     * Gets reset to 0 after every call to sync(). The reason for this setup is that
     * the multiple client calls to exchangeHalos() before sync() is called again
     * should get different MPI tags, because there is no global MPI_Barrier or MPI collective in between them.
     */
    mutable int haloEpoch_{0};
}

} // namespace cstone
