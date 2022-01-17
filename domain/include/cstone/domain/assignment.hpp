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
 * @brief Implementation of global particle assignment and distribution
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/domain/domaindecomp_mpi.hpp"
#include "cstone/domain/layout.hpp"
#include "cstone/tree/octree_internal.hpp"
#include "cstone/tree/octree_mpi.hpp"

#include "cstone/sfc/box_mpi.hpp"
#include "cstone/sfc/sfc.hpp"

namespace cstone
{

/*! @brief A class for global domain assignment and distribution
 *
 * @tparam KeyType  32- or 64-bit unsigned integer
 * @tparam T        float or double
 *
 * This class holds a low-res global octree which is replicated across all ranks and it presides over
 * the assignment of that tree to the ranks and performs the necessary point-2-point data exchanges
 * to send all particles to their owning ranks.
 */
template<class KeyType, class T>
class GlobalAssignment
{
public:
    GlobalAssignment(int rank, int nRanks, unsigned bucketSize, const Box<T>& box = Box<T>{0, 1})
        : myRank_(rank)
        , numRanks_(nRanks)
        , bucketSize_(bucketSize)
        , box_(box)
    {
        tree_       = std::vector<KeyType>{0, nodeRange<KeyType>(0)};
        nodeCounts_ = std::vector<unsigned>{bucketSize_ + 1};
    }

    /*! @brief Update the global tree
     *
     * @param[in]  particleStart_  first owned particle in x,y,z
     * @param[in]  particleEnd_    last owned particle in x,y,z
     * @param[in]  reorderFunctor  records the SFC order of the owned input coordinates
     * @param[out] particleKeys    will contain sorted particle SFC keys in the range [particleStart:particleEnd]
     * @param[in]  x               x coordinates
     * @param[in]  y               y coordinates
     * @param[in]  z               z coordinates
     * @return                     number of assigned particles
     *
     * This function does not modify / communicate any particle data.
     */
    template<class Tc, class Reorderer>
    LocalParticleIndex assign(LocalParticleIndex particleStart_, LocalParticleIndex particleEnd_,
                              Reorderer& reorderFunctor, KeyType* particleKeys, const Tc* x, const Tc* y, const Tc* z)
    {
        box_ = makeGlobalBox(x + particleStart_, x + particleEnd_, y + particleStart_, z + particleStart_, box_);

        // number of locally assigned particles to consider for global tree building
        LocalParticleIndex numParticles = particleEnd_ - particleStart_;

        gsl::span<KeyType> keyView(particleKeys + particleStart_, numParticles);

        // compute SFC particle keys only for particles participating in tree build
        computeSfcKeys(x + particleStart_, y + particleStart_, z + particleStart_,
                       sfcKindPointer(keyView.data()), numParticles, box_);

        // sort keys and keep track of ordering for later use
        reorderFunctor.setMapFromCodes(keyView.begin(), keyView.end());

        updateOctreeGlobal(keyView.begin(), keyView.end(), bucketSize_, tree_, nodeCounts_);

        if (firstCall_)
        {
            firstCall_ = false;
            // full build on first call
            while(!updateOctreeGlobal(keyView.begin(), keyView.end(), bucketSize_, tree_, nodeCounts_));
        }

        assignment_ = singleRangeSfcSplit(nodeCounts_, numRanks_);
        return assignment_.totalCount(myRank_);
    }

    /*! @brief Distribute particles to their assigned ranks based on previous assignment
     *
     * @param[in]    particleStart      first valid particle index before the exchange
     * @param[in]    particleEnd        last valid particle index before the exchange
     * @param[in]    bufferSize         size of particle buffers x,y,z and particleProperties
     * @param[inout] reorderFunctor     contains the ordering that accesses the range [particleStart:particleEnd]
     *                                  in SFC order
     * @param[out]   sfcOrder           If using the CPU reorderer, this is a duplicate copy. Otherwise provides
     *                                  the host space to download the ordering from the device.
     * @param[in]    keys               Sorted particle keys in [particleStart:particleEnd]
     * @param[inout] x                  particle x-coordinates
     * @param[inout] y                  particle y-coordinates
     * @param[inout] z                  particle z-coordinates
     * @param[inout] particleProperties remaining particle properties, h, m, etc.
     * @return                          index denoting the index range start of particles post-exchange
     *                                  plus a span with a view of the assigned particle keys
     *
     * Note: Instead of reordering the particle buffers right here after the exchange, we only keep track
     * of the reorder map that is required to transform the particle buffers into SFC-order. This allows us
     * to defer the reordering until we have done halo discovery. At that time, we know the final location
     * where to put the assigned particles inside the buffer, such that we can reorder directly to the final
     * location. This saves us from having to move around data inside the buffers for a second time.
     */
    template<class Reorderer, class Tc, class Th, class... Arrays>
    auto distribute(LocalParticleIndex particleStart,
                    LocalParticleIndex particleEnd,
                    LocalParticleIndex bufferSize,
                    Reorderer& reorderFunctor,
                    KeyType* keys,
                    Tc* x,
                    Tc* y,
                    Tc* z,
                    Th* h,
                    Arrays... particleProperties) const
    {
        LocalParticleIndex numParticles          = particleEnd - particleStart;
        LocalParticleIndex newNParticlesAssigned = assignment_.totalCount(myRank_);

        reallocate(numParticles, sfcOrder_);
        reorderFunctor.getReorderMap(sfcOrder_.data(), 0, numParticles);

        gsl::span<KeyType> keyView(keys + particleStart, numParticles);

        SendList domainExchangeSends = createSendList<KeyType>(assignment_, tree_, keyView);

        // Assigned particles are now inside the [particleStart:particleEnd] range, but not exclusively.
        // Leftover particles from the previous step can also be contained in the range.
        auto [newStart, newEnd] =
            exchangeParticles(domainExchangeSends, myRank_, particleStart, particleEnd, bufferSize,
                              newNParticlesAssigned, sfcOrder_.data(), x, y, z, h, particleProperties...);

        LocalParticleIndex envelopeSize = newEnd - newStart;
        keyView                         = gsl::span<KeyType>(keys + newStart, envelopeSize);

        computeSfcKeys(x + newStart, y + newStart, z + newStart, sfcKindPointer(keyView.begin()), envelopeSize, box_);
        // sort keys and keep track of the ordering
        reorderFunctor.setMapFromCodes(keyView.begin(), keyView.end());

        // thanks to the sorting, we now know the exact range of the assigned particles:
        // [newStart + offset, newStart + offset + newNParticlesAssigned]
        LocalParticleIndex offset = findNodeAbove<KeyType>(keyView, tree_[assignment_.firstNodeIdx(myRank_)]);
        // restrict the reordering to take only the assigned particles into account and ignore the others in
        // [newStart:newEnd]
        reorderFunctor.restrictRange(offset, newNParticlesAssigned);
        reorderFunctor(h + newStart, h);

        return std::make_tuple(newStart, gsl::span<const KeyType>{keys + newStart + offset, newNParticlesAssigned});
    }

    std::vector<int> findPeers(float theta)
    {
        Octree<KeyType> domainTree;
        domainTree.update(tree_.begin(), tree_.end());
        std::vector<int> peers = findPeersMac(myRank_, assignment_, domainTree, box_, theta);
        return peers;
    }

    //! @brief read only visibility of the global octree leaves to the outside
    gsl::span<const KeyType> tree() const { return tree_; }

    //! @brief read only visibility of the global octree leaf counts to the outside
    gsl::span<const unsigned> nodeCounts() const { return nodeCounts_; }

    //! @brief the global coordinate bounding box
    const Box<T>& box() const { return box_; }

    //! @brief return the space filling curve rank assignment
    const SpaceCurveAssignment& assignment() const { return assignment_; }

private:
    int myRank_;
    int numRanks_;
    unsigned bucketSize_;

    //! @brief global coordinate bounding box
    Box<T> box_;

    SpaceCurveAssignment assignment_;

    //! @brief storage for downloading the sfc ordering from the GPU
    mutable std::vector<LocalParticleIndex> sfcOrder_;

    //! @brief cornerstone tree leaves for global domain decomposition
    std::vector<KeyType> tree_;
    std::vector<unsigned> nodeCounts_;

    bool firstCall_{true};
};

} // namespace cstone
