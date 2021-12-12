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
#include "cstone/traversal/collisions.hpp"
#include "cstone/traversal/peers.hpp"

#include "cstone/gravity/treewalk.hpp"
#include "cstone/gravity/upsweep.hpp"
#include "cstone/halos/exchange_halos.hpp"

#include "cstone/tree/octree_mpi.hpp"
#include "cstone/focus/octree_focus_mpi.hpp"

#include "cstone/sfc/box_mpi.hpp"

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
class GlobalTree
{
public:
    GlobalTree(int rank, int nRanks, unsigned bucketSize, const Box<T>& box = Box<T>{0, 1})
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
     * @param[in]    particleStart_        first valid particle index before the exchange
     * @param[in]    particleEnd_          last valid particle index before the exchange
     * @param[in]    bufferSize            size of particle buffers x,y,z and particleProperties
     * @param[inout] reorderFunctor        contains the ordering that accesses the range [particleStart:particleEnd]
     *                                     in SFC order
     * @param[out]   sfcOrder              If using the CPU reorderer, this is a duplicate copy. Otherwise provides
     *                                     the host space to download the ordering from the device.
     * @param[in]    particleKeys          Sorted particle keys in [particleStart:particleEnd]
     * @param[inout] x                     particle x-coordinates
     * @param[inout] y                     particle y-coordinates
     * @param[inout] z                     particle z-coordinates
     * @param[inout] particleProperties    remaining particle properties, h, m, etc.
     * @return                             index pair denoting the index range of particles post-exchange
     *                                     plus the number of particles from before the exchange that have
     *                                     a lower SFC key than the first assigned particle
     */
    template<class Reorderer, class Tc, class... Arrays>
    std::tuple<LocalParticleIndex, LocalParticleIndex, LocalParticleIndex>
    distribute(LocalParticleIndex particleStart_,
               LocalParticleIndex particleEnd_,
               LocalParticleIndex bufferSize,
               Reorderer& reorderFunctor,
               LocalParticleIndex* sfcOrder,
               KeyType* particleKeys,
               Tc* x,
               Tc* y,
               Tc* z,
               Arrays... particleProperties) const
    {
        LocalParticleIndex numParticles          = particleEnd_ - particleStart_;
        LocalParticleIndex newNParticlesAssigned = assignment_.totalCount(myRank_);

        reorderFunctor.getReorderMap(sfcOrder);

        gsl::span<KeyType> keyView(particleKeys + particleStart_, numParticles);

        SendList domainExchangeSends = createSendList<KeyType>(assignment_, tree_, keyView);

        std::tie(particleStart_, particleEnd_) =
            exchangeParticles<T>(domainExchangeSends, myRank_, particleStart_, particleEnd_, bufferSize,
                                 newNParticlesAssigned, sfcOrder, x, y, z, particleProperties...);

        numParticles = particleEnd_ - particleStart_;
        keyView      = gsl::span<KeyType>(particleKeys + particleStart_, numParticles);

        // refresh particleKeys and ordering
        computeSfcKeys(x + particleStart_, y + particleStart_, z + particleStart_, sfcKindPointer(keyView.begin()),
                       numParticles, box_);
        reorderFunctor.setMapFromCodes(keyView.begin(), keyView.end());
        reorderFunctor.getReorderMap(sfcOrder);

        LocalParticleIndex compactOffset = findNodeAbove<KeyType>(keyView, tree_[assignment_.firstNodeIdx(myRank_)]);

        return {particleStart_, particleEnd_, compactOffset};
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

    //! @brief cornerstone tree leaves for global domain decomposition
    std::vector<KeyType> tree_;
    std::vector<unsigned> nodeCounts_;

    bool firstCall_{true};
};

template<class KeyType, class T, class Accelerator = CpuTag>
class Domain
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
     * @param theta           angle parameter to control focus resolution and gravity accuracy
     * @param box             global bounding box, default is non-pbc box
     *                        for each periodic dimension in @a box, the coordinate min/max
     *                        limits will never be changed for the lifetime of the Domain
     *
     */
    Domain(int rank,
           int nRanks,
           unsigned bucketSize,
           unsigned bucketSizeFocus,
           float theta,
           const Box<T>& box = Box<T>{0, 1})
        : myRank_(rank)
        , numRanks_(nRanks)
        , bucketSizeFocus_(bucketSizeFocus)
        , theta_(theta)
        , focusedTree_(bucketSizeFocus_, theta_)
        , globalTree_(rank, nRanks, bucketSize, box)
    {
        if (bucketSize < bucketSizeFocus_)
        {
            throw std::runtime_error("The bucket size of the global tree must not be smaller than the bucket size"
                                     " of the focused tree\n");
        }
    }

    /*! @brief Domain update sequence for particles with coordinates x,y,z, interaction radius h and their properties
     *
     * @param[inout] x             floating point coordinates
     * @param[inout] y
     * @param[inout] z
     * @param[inout] h             interaction radii in SPH convention, actual interaction radius
     *                             is twice the value in h
     * @param[out]   particleKeys  SFC particleKeys
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
     *   - Content of particleKeys is irrelevant as it will be resized to fit x,y,z,h and particleProperties
     *
     * ============================================================================================================
     * Postconditions:
     * ============================================================================================================
     *
     *   Array sizes:
     *   ------------
     *   - All arrays, x,y,z,h, particleKeys and particleProperties are resized with space for the newly assigned
     *     particles AND their halos.
     *
     *   Content of x,y,z and h
     *   ----------------------------
     *   - x,y,z,h at indices from startIndex() to endIndex() contain assigned particles that the executing rank owns,
     *     all other elements are _halos_ of the assigned particles, i.e. the halos for x,y,z,h and particleKeys are
     *     already in place post-call.
     *
     *   Content of particleProperties
     *   ----------------------------
     *   - particleProperties arrays contain the updated properties at indices from startIndex() to endIndex(),
     *     i.e. index i refers to a property of the particle with coordinates (x[i], y[i], z[i]).
     *     Content of elements outside this range is _undefined_, but can be filled with the corresponding halo data
     *     by a subsequent call to exchangeHalos(particleProperty), such that also for i outside
     *     [startIndex():endIndex()], particleProperty[i] is a property of the halo particle with
     *     coordinates (x[i], y[i], z[i]).
     *
     *   Content of particleKeys
     *   ----------------
     *   - The particleKeys output is sorted and contains the Morton particleKeys of assigned _and_ halo particles,
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
     *      7. resize x,y,z,h,particleKeys and properties to new number of assigned + halo particles
     *      8. exchange coordinates, h, and properties of assigned particles
     *      9. morton sort exchanged assigned particles
     *     10. exchange halo particles
     */
    template<class... Vectors>
    void sync(std::vector<T>& x, std::vector<T>& y, std::vector<T>& z, std::vector<T>& h,
              std::vector<KeyType>& particleKeys,
              Vectors&... particleProperties)
    {
        // bounds initialization on first call, use all particles
        if (firstCall_)
        {
            particleStart_ = 0;
            particleEnd_   = x.size();
            layout_        = {0, LocalParticleIndex(x.size())};
        }

        if (!sizesAllEqualTo(layout_.back(), particleKeys, x, y, z, h, particleProperties...))
        {
            throw std::runtime_error("Domain sync: input array sizes are inconsistent\n");
        }

        haloEpoch_ = 0;

        /* SFC decomposition phase *********************************************************/

        LocalParticleIndex newNParticlesAssigned = globalTree_.assign(
            particleStart_, particleEnd_, reorderFunctor, particleKeys.data(), x.data(), y.data(), z.data());

        /* Domain particles update phase *********************************************************/

        if (newNParticlesAssigned > x.size())
        {
            reallocate(newNParticlesAssigned, particleKeys, x, y, z, h, particleProperties...);
        }
        std::vector<LocalParticleIndex> sfcOrder(x.size());

        LocalParticleIndex compactOffset;
        std::tie(particleStart_, particleEnd_, compactOffset) = globalTree_.distribute(
            particleStart_, particleEnd_, x.size(), reorderFunctor, sfcOrder.data(), particleKeys.data(), x.data(),
            y.data(), z.data(), h.data(), particleProperties.data()...);

        // the range [particleStart_:particleEnd_] can still contain leftover particles from the previous step
        // but [particleStart_ + compactOffset : particleStart_ + compactOffset + newNParticlesAssigned]
        // exclusively refers to locally assigned particles in SFC order when accessed through sfcOrder
        gsl::span<KeyType> keyView(particleKeys.data() + particleStart_ + compactOffset, newNParticlesAssigned);

        Box<T> box                             = globalTree_.box();
        const SpaceCurveAssignment& assignment = globalTree_.assignment();
        gsl::span<const KeyType> globalTree    = globalTree_.tree();
        gsl::span<const unsigned> globalCounts = globalTree_.nodeCounts();

        /* Focus tree update phase *********************************************************/

        Octree<KeyType> domainTree;
        domainTree.update(globalTree.begin(), globalTree.end());
        std::vector<int> peers = findPeersMac(myRank_, assignment, domainTree, box, theta_);

        focusedTree_.update(box, keyView, myRank_, peers, assignment, globalTree, globalCounts);
        if (firstCall_)
        {
            // we must not call updateGlobal again before all ranks have completed the previous call,
            // otherwise point-2-point messages from different updateGlobal calls can get mixed up
            MPI_Barrier(MPI_COMM_WORLD);
            int converged = 0;
            while (converged != numRanks_)
            {
                converged = focusedTree_.update(box, keyView, myRank_, peers, assignment, globalTree, globalCounts);
                MPI_Allreduce(MPI_IN_PLACE, &converged, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            }
            firstCall_ = false;
        }

        std::vector<TreeIndexPair> focusAssignment
            = translateAssignment<KeyType>(assignment, globalTree, focusedTree_.treeLeaves(), peers, myRank_);

        /* Halo discovery phase *********************************************************/

        std::vector<float> haloRadii(nNodes(focusedTree_.treeLeaves()));
        computeHaloRadii(focusedTree_.treeLeaves().data(),
                         nNodes(focusedTree_.treeLeaves()),
                         {keyView.data(), keyView.size()},
                         sfcOrder.data() + compactOffset,
                         h.data() + particleStart_,
                         haloRadii.data());

        std::vector<int> haloFlags(nNodes(focusedTree_.treeLeaves()), 0);
        findHalos(focusedTree_.octree(),
                  haloRadii.data(),
                  box,
                  focusAssignment[myRank_].start(),
                  focusAssignment[myRank_].end(),
                  haloFlags.data());

        /* Compute new layout *********************************************************/

        reallocate(nNodes(focusedTree_.treeLeaves()) + 1, layout_);
        computeNodeLayout(focusedTree_.leafCounts(), haloFlags, focusAssignment[myRank_].start(),
                          focusAssignment[myRank_].end(), layout_);
        auto newParticleStart = layout_[focusAssignment[myRank_].start()];
        auto newParticleEnd   = layout_[focusAssignment[myRank_].end()];
        auto numParticles     = layout_.back();

        outgoingHaloIndices_
            = exchangeRequestKeys<KeyType>(focusedTree_.treeLeaves(), haloFlags,
                                           keyView, newParticleStart, focusAssignment, peers);
        checkIndices(outgoingHaloIndices_, newParticleStart, newParticleEnd);

        incomingHaloIndices_ = computeHaloReceiveList(layout_, haloFlags, focusAssignment, peers);

        /* Rearrange particle buffers *********************************************************/

        reallocate(numParticles, x, y, z, h, particleProperties...);

        auto reorderArray = [this, newParticleStart, compactOffset, newNParticlesAssigned](auto ptr)
        {
            reorderFunctor(ptr + this->particleStart_, ptr + newParticleStart, compactOffset, newNParticlesAssigned);
        };
        std::tuple particleArrays{x.data(), y.data(), z.data(), h.data(), particleProperties.data()...};
        for_each_tuple(reorderArray, particleArrays);

        std::vector<KeyType> newKeys(numParticles);
        std::copy(keyView.begin(), keyView.end(), newKeys.begin() + newParticleStart);
        swap(particleKeys, newKeys);

        particleStart_ = newParticleStart;
        particleEnd_   = newParticleEnd;

        /* Halo exchange phase *********************************************************/

        exchangeHalos(x, y, z, h);

        // compute SFC keys of received halo particles
        computeSfcKeys(x.data(), y.data(), z.data(), sfcKindPointer(particleKeys.data()),
                       particleStart_, box);
        computeSfcKeys(x.data() + particleEnd_, y.data() + particleEnd_, z.data() + particleEnd_,
                       sfcKindPointer(particleKeys.data()) + particleEnd_, x.size() - particleEnd_, box);
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
        if (!sizesAllEqualTo(layout_.back(), arrays...))
        {
            throw std::runtime_error("halo exchange array sizes inconsistent with previous sync operation\n");
        }

        haloexchange(haloEpoch_++, incomingHaloIndices_, outgoingHaloIndices_, arrays.data()...);
    }

    /*! @brief compute gravitational accelerations
     *
     * @param[in]    x    x-coordinates
     * @param[in]    y    y-coordinates
     * @param[in]    z    z-coordinates
     * @param[in]    h    smoothing lengths
     * @param[in]    m    particle masses
     * @param[in]    G    gravitational constant
     * @param[inout] ax   x-acceleration to add to
     * @param[inout] ay   y-acceleration to add to
     * @param[inout] az   z-acceleration to add to
     * @return            total gravitational potential energy
     */
    T addGravityAcceleration(gsl::span<const T> x, gsl::span<const T> y, gsl::span<const T> z, gsl::span<const T> h,
                             gsl::span<const T> m, float G, gsl::span<T> ax, gsl::span<T> ay, gsl::span<T> az)
    {
        const Octree<KeyType>& octree = focusedTree_.octree();
        std::vector<GravityMultipole<T>> multipoles(octree.numTreeNodes());
        computeMultipoles(octree, layout_, x.data(), y.data(), z.data(), m.data(), multipoles.data());

        return computeGravity(octree, multipoles.data(), layout_.data(), 0, octree.numLeafNodes(),
                              x.data(), y.data(), z.data(), h.data(), m.data(), globalTree_.box(), theta_,
                              G, ax.data(), ay.data(), az.data());
    }

    //! @brief return the index of the first particle that's part of the local assignment
    [[nodiscard]] LocalParticleIndex startIndex() const { return particleStart_; }

    //! @brief return one past the index of the last particle that's part of the local assignment
    [[nodiscard]] LocalParticleIndex endIndex() const   { return particleEnd_; }

    //! @brief return number of locally assigned particles
    [[nodiscard]] LocalParticleIndex nParticles() const { return endIndex() - startIndex(); }

    //! @brief return number of locally assigned particles plus number of halos
    [[nodiscard]] LocalParticleIndex nParticlesWithHalos() const { return layout_.back(); }

    //! @brief read only visibility of the global octree leaves to the outside
    gsl::span<const KeyType> tree() const { return globalTree_.tree(); }

    //! @brief read only visibility of the focused octree leaves to the outside
    gsl::span<const KeyType> focusedTree() const { return focusedTree_.treeLeaves(); }

    //! @brief return the coordinate bounding box from the previous sync call
    Box<T> box() const { return globalTree_.box(); }

private:

    //! @brief check that only owned particles in [particleStart_:particleEnd_] are sent out as halos
    void checkIndices(const SendList& sendList, LocalParticleIndex start, LocalParticleIndex end)
    {
        for (const auto& manifest : sendList)
        {
            for (size_t ri = 0; ri < manifest.nRanges(); ++ri)
            {
                assert(!overlapTwoRanges(LocalParticleIndex{0}, start,
                                         manifest.rangeStart(ri), manifest.rangeEnd(ri)));
                assert(!overlapTwoRanges(end, layout_.back(),
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
    int numRanks_;
    unsigned bucketSizeFocus_;

    //! @brief MAC parameter for focus resolution and gravity treewalk
    float theta_;

    /*! @brief array index of first local particle belonging to the assignment
     *  i.e. the index of the first particle that belongs to this rank and is not a halo.
     */
    LocalParticleIndex particleStart_{0};
    //! @brief index (upper bound) of last particle that belongs to the assignment
    LocalParticleIndex particleEnd_{0};

    SendList incomingHaloIndices_;
    SendList outgoingHaloIndices_;

    /*! @brief locally focused, fully traversable octree, used for halo discovery and exchange
     *
     * -Uses bucketSizeFocus_ as the maximum particle count per leaf within the focused SFC area.
     * -Outside the focus area, each leaf node with a particle count larger than bucketSizeFocus_
     *  fulfills a MAC with theta as the opening parameter
     * -Also contains particle counts.
     */
    FocusedOctree<KeyType> focusedTree_;

    GlobalTree<KeyType, T> globalTree_;

    //! @brief particle offsets of each leaf node in focusedTree_, length = focusedTree_.treeLeaves().size()
    std::vector<LocalParticleIndex> layout_;

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
