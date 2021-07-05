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
 * @brief A domain class to manage distributed particles and their halos.
 *
 * Particles are represented by x,y,z coordinates, interaction radii and
 * a user defined number of additional properties, such as masses or charges.
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/domain/domaindecomp_mpi.hpp"
#include "cstone/domain/domain_traits.hpp"
#include "cstone/domain/exchange_keys.hpp"
#include "cstone/domain/layout.hpp"
#include "cstone/domain/peers.hpp"

#include "cstone/halos/discovery.hpp"
#include "cstone/halos/exchange_halos.hpp"

#include "cstone/tree/octree_mpi.hpp"
#include "cstone/tree/octree_focus_mpi.hpp"

#include "cstone/sfc/box_mpi.hpp"

namespace cstone
{

template<class KeyType, class T, class Accelerator = CpuTag>
class FocusedDomain
{
    static_assert(std::is_unsigned<KeyType>{}, "SFC key type needs to be an unsigned integer\n");

    using ReorderFunctor = ReorderFunctor_t<Accelerator, T, KeyType, LocalParticleIndex>;

public:
    /*! @brief construct empty Domain
     *
     * @param rank            executing rank
     * @param nRanks          number of ranks
     * @param bucketSize      build global tree for domain decomposition with max @a bucketSize particles per node
     * @param bucketSizeFocus maximum number of particles per leaf node inside the assigned part of the SFC
     * @param box             global bounding box, default is non-pbc box
     *                        for each periodic dimension in @a box, the coordinate min/max
     *                        limits will never be changed for the lifetime of the Domain
     *
     */
    explicit FocusedDomain(int rank, int nRanks, unsigned bucketSize, unsigned bucketSizeFocus,
                           const Box<T>& box = Box<T>{0,1})
        : myRank_(rank), nRanks_(nRanks), bucketSize_(bucketSize), bucketSizeFocus_(bucketSizeFocus), box_(box),
          focusedTree_(bucketSizeFocus_, theta_)
    {
        if (bucketSize_ < bucketSizeFocus_)
        {
            throw std::runtime_error("The bucket size of the global tree must not be smaller than the bucket size"
                                     " of the focused tree\n");
        }
    }

    /*! @brief Domain update sequence for particles with coordinates x,y,z, interaction radius h and their properties
     *
     * @param[inout] x      floating point coordinates
     * @param[inout] y
     * @param[inout] z
     * @param[inout] h      interaction radii in SPH convention, actual interaction radius is twice the value in h
     * @param[out]   codes  Morton codes
     *
     * @param[inout] particleProperties  particle properties to distribute along with the coordinates
     *                                   e.g. mass or charge
     *
     * ============================================================================================================
     * Preconditions:
     * ============================================================================================================
     *
     *   - Array sizes of x,y,z,h and particleProperties are identical
     *     AND equal to the internally stored value of localNParticles_ from the previous call, except
     *     on the first call. This is checked.
     *
     *     This means that none of the argument arrays can be resized between calls of this function.
     *     Or in other words, particles cannot be created or destroyed.
     *     (If this should ever be required though, it can be easily enabled by allowing the assigned
     *     index range from startIndex() to endIndex() to be modified from the outside.)
     *
     *   - The particle order is irrelevant
     *
     *   - Content of codes is irrelevant as it will be resized to fit x,y,z,h and particleProperties
     *
     * ============================================================================================================
     * Postconditions:
     * ============================================================================================================
     *
     *   Array sizes:
     *   ------------
     *   - All arrays, x,y,z,h, codes and particleProperties are resized with space for the newly assigned particles
     *     AND their halos.
     *
     *   Content of x,y,z and h
     *   ----------------------------
     *   - x,y,z,h at indices from startIndex() to endIndex() contain assigned particles that the executing rank owns,
     *     all other elements are _halos_ of the assigned particles, i.e. the halos for x,y,z,h and codes are already
     *     in place post-call.
     *
     *   Content of particleProperties
     *   ----------------------------
     *   - particleProperties arrays contain the updated properties at indices from startIndex() to endIndex(),
     *     i.e. index i refers to a property of the particle with coordinates (x[i], y[i], z[i]).
     *     Content of elements outside this range is _undefined_, but can be filled with the corresponding halo data
     *     by a subsequent call to exchangeHalos(particleProperty), such that also for i outside [startIndex():endIndex()],
     *     particleProperty[i] is a property of the halo particle with coordinates (x[i], y[i], z[i]).
     *
     *   Content of codes
     *   ----------------
     *   - The codes output is sorted and contains the Morton codes of assigned _and_ halo particles,
     *     i.e. all arrays will be output in Morton order.
     *
     *   Internal state of the domain
     *   ----------------------------
     *   The following members are modified by calling this function:
     *   - Update of the global octree, for use as starting guess in the next call
     *   - Update of the assigned range startIndex() and endIndex()
     *   - Update of the total local particle count, i.e. assigned + halo particles
     *   - Update of the halo exchange patterns, for subsequent use in exchangeHalos
     *   - Update of the global coordinate bounding box
     *
     * ============================================================================================================
     * Update sequence:
     * ============================================================================================================
     *      1. compute global coordinate bounding box
     *      2. compute global octree
     *      3. compute max_h per octree node
     *      4. assign octree to ranks
     *      5. discover halos
     *      6. compute particle layout, i.e. count number of halos and assigned particles
     *         and compute halo send and receive index ranges
     *      7. resize x,y,z,h,codes and properties to new number of assigned + halo particles
     *      8. exchange coordinates, h, and properties of assigned particles
     *      9. morton sort exchanged assigned particles
     *     10. exchange halo particles
     */
    template<class... Vectors>
    void sync(std::vector<T>& x, std::vector<T>& y, std::vector<T>& z, std::vector<T>& h, std::vector<KeyType>& codes,
              Vectors&... particleProperties)
    {
        // bounds initialization on first call, use all particles
        if (firstCall_)
        {
            particleStart_   = 0;
            particleEnd_     = x.size();
            localNParticles_ = x.size();
            tree_       = std::vector<KeyType>{0, nodeRange<KeyType>(0)};
            nodeCounts_ = std::vector<unsigned>{localNParticles_};
        }

        if (!sizesAllEqualTo(localNParticles_, x, y, z, h, particleProperties...))
        {
            throw std::runtime_error("Domain sync: input array sizes are inconsistent\n");
        }

        haloEpoch_ = 0;

        /* SFC decomposition phase *********************************************************/

        box_ = makeGlobalBox(cbegin(x) + particleStart_, cbegin(x) + particleEnd_,
                             cbegin(y) + particleStart_,
                             cbegin(z) + particleStart_, box_);

        // number of locally assigned particles to consider for global tree building
        LocalParticleIndex numParticles = particleEnd_ - particleStart_;

        codes.resize(numParticles);

        // compute morton codes only for particles participating in tree build
        computeMortonCodes(cbegin(x) + particleStart_, cbegin(x) + particleEnd_,
                           cbegin(y) + particleStart_,
                           cbegin(z) + particleStart_,
                           begin(codes), box_);

        // reorder the codes according to the ordering
        // has the same net effect as std::sort(begin(mortonCodes), end(mortonCodes)),
        // but with the difference that we explicitly know the ordering, such
        // that we can later apply it to the x,y,z,h arrays or to access them in the Morton order
        reorderFunctor.setMapFromCodes(codes.data(), codes.data() + codes.size());

        // extract ordering for use in e.g. exchange particles
        std::vector<LocalParticleIndex> mortonOrder(numParticles);
        reorderFunctor.getReorderMap(mortonOrder.data());

        // compute the global octree in cornerstone format (leaves only)
        // the resulting tree and node counts will be identical on all ranks
        updateOctreeGlobal(codes.data(), codes.data() + numParticles, bucketSize_, tree_, nodeCounts_);

        if (firstCall_)
        {
            // full build on first call
            while(!updateOctreeGlobal(codes.data(), codes.data() + numParticles, bucketSize_, tree_, nodeCounts_));
        }

        // assign one single range of Morton codes each rank
        SpaceCurveAssignment assignment = singleRangeSfcSplit(nodeCounts_, nRanks_);
        LocalParticleIndex newNParticlesAssigned = assignment.totalCount(myRank_);

        //if (myRank_ == 0)
        //{
        //    std::cout << "assignment ";
        //    for (int rank = 0; rank < nRanks_; ++rank)
        //        std::cout << std::oct << tree_[assignment.firstNodeIdx(rank)] << " ";
        //    std::cout << tree_[assignment.lastNodeIdx(nRanks_-1)] << std::dec << std::endl;
        //}

        /* Domain particles update phase *********************************************************/

        // compute send array ranges for domain exchange
        // index ranges in domainExchangeSends are valid relative to the sorted code array mortonCodes
        // note that there is no offset applied to mortonCodes, because it was constructed
        // only with locally assigned particles
        SendList domainExchangeSends = createSendList<KeyType>(assignment, tree_, codes);

        // resize arrays to new sizes
        reallocate(newNParticlesAssigned, x,y,z,h, particleProperties...);
        reallocate(newNParticlesAssigned, codes);
        // exchange assigned particles
        exchangeParticles<T>(domainExchangeSends, Rank(myRank_), newNParticlesAssigned,
                             particleStart_, LocalParticleIndex(0), mortonOrder.data(),
                             x.data(), y.data(), z.data(), h.data(), particleProperties.data()...);
        // recompute SFC codes
        computeMortonCodes(begin(x), end(x), begin(y), begin(z), begin(codes), box_);
        // sort codes and update reorder-map inside the functor
        reorderFunctor.setMapFromCodes(codes.data(), codes.data() + codes.size());
        {
            std::array<std::vector<T>*, 4 + sizeof...(Vectors)> particleArrays{&x, &y, &z, &h, &particleProperties...};
            for (std::size_t i = 0; i < particleArrays.size(); ++i)
            {
                //reorderFunctor(particleArrays[i]->data() + particleStart_) ;
                reorderFunctor(particleArrays[i]->data()) ;
            }
        }

        /* Focus tree update phase *********************************************************/

        Octree<KeyType> domainTree;
        domainTree.update(begin(tree_), end(tree_));
        std::vector<int> peers = findPeersMac(myRank_, assignment, domainTree, box_, theta_);

        int converged = focusedTree_.updateGlobal(box_, codes, myRank_, peers, assignment, tree_, nodeCounts_);
        if (firstCall_)
        {
            MPI_Allreduce(MPI_IN_PLACE, &converged, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            //int cnt = 1;
            while (converged != nRanks_)
            {
                converged = focusedTree_.updateGlobal(box_, codes, myRank_, peers, assignment, tree_, nodeCounts_);
                MPI_Allreduce(MPI_IN_PLACE, &converged, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
                //cnt++;
            }
            //if (myRank_ == 0) { std::cout << "focus converged in " << cnt << std::endl; }
            firstCall_ = false;
        }

        std::vector<TreeIndexPair> focusAssignment
            = translateAssignment<KeyType>(assignment, tree_, focusedTree_.treeLeaves(), peers, myRank_);

        /* Halo discovery phase *********************************************************/

        mortonOrder.resize(newNParticlesAssigned);
        std::iota(begin(mortonOrder), end(mortonOrder), LocalParticleIndex(0));

        std::vector<float> haloRadii(nNodes(focusedTree_.treeLeaves()));
        computeHaloRadii(focusedTree_.treeLeaves().data(),
                         nNodes(focusedTree_.treeLeaves()),
                         codes.data(),
                         codes.data() + codes.size(),
                         mortonOrder.data(),
                         h.data(),
                         haloRadii.data());

        std::vector<int> haloFlags(nNodes(focusedTree_.treeLeaves()), 0);
        findHalos<KeyType, float>(focusedTree_.treeLeaves(),
                                  focusedTree_.binaryTree(),
                                  haloRadii,
                                  box_,
                                  focusAssignment[myRank_].start(),
                                  focusAssignment[myRank_].end(),
                                  haloFlags.data());

        /* Halo exchange phase *********************************************************/

        std::vector<LocalParticleIndex> layout = computeNodeLayout(focusedTree_.leafCounts(), haloFlags,
                                                                   focusAssignment[myRank_].start(),
                                                                   focusAssignment[myRank_].end());
        localNParticles_ = layout.back();
        particleStart_   = layout[focusAssignment[myRank_].start()];
        particleEnd_     = particleStart_ + newNParticlesAssigned;

        outgoingHaloIndices_
            = exchangeRequestKeys<KeyType>(focusedTree_.treeLeaves(), haloFlags,
                                           gsl::span<const KeyType>(codes.data(), newNParticlesAssigned),
                                           particleStart_, focusAssignment, peers);
        checkIndices(outgoingHaloIndices_);

        incomingHaloIndices_ = computeHaloReceiveList(layout, haloFlags, focusAssignment, peers);

        relocate(localNParticles_, particleStart_, x, y, z, h, particleProperties...);
        relocate(localNParticles_, particleStart_, codes);

        exchangeHalos(x, y, z, h);

        // compute SFC keys of received halo particles
        computeMortonCodes(cbegin(x), cbegin(x) + particleStart_,
                           cbegin(y),
                           cbegin(z),
                           begin(codes), box_);
        computeMortonCodes(cbegin(x) + particleEnd_, cend(x),
                           cbegin(y) + particleEnd_,
                           cbegin(z) + particleEnd_,
                           begin(codes) + particleEnd_, box_);
    }

    /*! @brief repeat the halo exchange pattern from the previous sync operation for a different set of arrays
     *
     * @param[inout] arrays  std::vector<float or double> of size localNParticles_
     *
     * Arrays are not resized or reallocated. This is used e.g. for densities.
     * Note: function is const, but modiefies mutable haloEpoch_ counter.
     */
    template<class...Arrays>
    void exchangeHalos(Arrays&... arrays) const
    {
        if (!sizesAllEqualTo(localNParticles_, arrays...))
        {
            throw std::runtime_error("halo exchange array sizes inconsistent with previous sync operation\n");
        }

        haloexchange<T>(haloEpoch_++, incomingHaloIndices_, outgoingHaloIndices_, arrays.data()...);
    }

    //! @brief return the index of the first particle that's part of the local assignment
    [[nodiscard]] LocalParticleIndex startIndex() const { return particleStart_; }

    //! @brief return one past the index of the last particle that's part of the local assignment
    [[nodiscard]] LocalParticleIndex endIndex() const   { return particleEnd_; }

    //! @brief return number of locally assigned particles
    [[nodiscard]] LocalParticleIndex nParticles() const { return endIndex() - startIndex(); }

    //! @brief return number of locally assigned particles plus number of halos
    [[nodiscard]] LocalParticleIndex nParticlesWithHalos() const { return localNParticles_; }

    //! @brief read only visibility of the global octree leaves to the outside
    gsl::span<const KeyType> tree() const { return tree_; }

    //! @brief read only visibility of the focused octree leaves to the outside
    gsl::span<const KeyType> focusedTree() const { return focusedTree_.treeLeaves(); }

    //! @brief return the coordinate bounding box from the previous sync call
    Box<T> box() const { return box_; }

private:

    //! @brief check that only owned particles in [particleStart_:particleEnd_] are sent out as halos
    void checkIndices(const SendList& sendList)
    {
        for (const auto& manifest : sendList)
        {
            for (size_t ri = 0; ri < manifest.nRanges(); ++ri)
            {
                assert(!overlapTwoRanges(LocalParticleIndex{0}, particleStart_,
                                         manifest.rangeStart(ri), manifest.rangeEnd(ri)));
                assert(!overlapTwoRanges(particleEnd_, localNParticles_,
                                         manifest.rangeStart(ri), manifest.rangeEnd(ri)));
            }
        }
    }

    //! @brief return true if all array sizes are equal to value
    template<class... Arrays>
    static bool sizesAllEqualTo(std::size_t value, Arrays&... arrays)
    {
        std::array<std::size_t, sizeof...(Arrays)> sizes{arrays.size()...};
        return std::count(begin(sizes), end(sizes), value) == sizes.size();
    }

    int myRank_;
    int nRanks_;
    unsigned bucketSize_;
    unsigned bucketSizeFocus_;

    /*! @brief array index of first local particle belonging to the assignment
     *  i.e. the index of the first particle that belongs to this rank and is not a halo.
     */
    LocalParticleIndex particleStart_{0};
    //! @brief index (upper bound) of last particle that belongs to the assignment
    LocalParticleIndex particleEnd_{0};
    //! @brief number of locally present particles, = number of halos + assigned particles
    LocalParticleIndex localNParticles_{0};

    //! @brief coordinate bounding box, each non-periodic dimension is at a sync call
    Box<T> box_;

    SendList incomingHaloIndices_;
    SendList outgoingHaloIndices_;

    //! @brief cornerstone tree leaves for global domain decomposition
    std::vector<KeyType> tree_;
    std::vector<unsigned> nodeCounts_;

    float theta_{1.0};

    /*! @brief locally focused, fully traversable octree, used for halo discovery and exchange
     *
     * -Uses bucketSizeFocus_ as the maximum particle count per leaf within the focused SFC area.
     * -Outside the focus area, each leaf node with a particle count larger than bucketSizeFocus_
     *  fulfills a MAC with theta as the opening parameter
     * -Also contains particle counts.
     */
    FocusedOctree<KeyType> focusedTree_;

    bool firstCall_{true};

    /*! @brief counter for halo exchange calls between sync() calls
     *
     * Gets reset to 0 after every call to sync(). The reason for this setup is that
     * the multiple client calls to exchangeHalos() before sync() is called again
     * should get different MPI tags, because there is no global MPI_Barrier or MPI collective in between them.
     */
     mutable int haloEpoch_{0};

    ReorderFunctor reorderFunctor;
};

} // namespace cstone
