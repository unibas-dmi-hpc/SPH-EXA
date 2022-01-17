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

#include "cstone/domain/assignment.hpp"
#include "cstone/domain/domain_traits.hpp"
#include "cstone/domain/exchange_keys.hpp"
#include "cstone/domain/layout.hpp"
#include "cstone/traversal/collisions.hpp"
#include "cstone/traversal/peers.hpp"

#include "cstone/gravity/treewalk.hpp"
#include "cstone/gravity/upsweep.hpp"
#include "cstone/halos/exchange_halos.hpp"
#include "cstone/halos/halos.hpp"

#include "cstone/focus/octree_focus_mpi.hpp"

#include "cstone/sfc/box_mpi.hpp"

namespace cstone
{

template<class KeyType, class T, class Accelerator = CpuTag>
class Domain
{
    static_assert(std::is_unsigned<KeyType>{}, "SFC key type needs to be an unsigned integer\n");

    using ReorderFunctor = ReorderFunctor_t<Accelerator, T, KeyType, LocalIndex>;

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
        , focusTree_(rank, nRanks, bucketSizeFocus_, theta_)
        , global_(rank, nRanks, bucketSize, box)
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
     *   - The particleKeys output is sorted and contains the SFC particleKeys of assigned _and_ halo particles,
     *     i.e. all arrays will be output in SFC order.
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
     *      9. SFC sort exchanged assigned particles
     *     10. exchange halo particles
     */
    template<class... Vectors>
    void sync(std::vector<KeyType>& particleKeys,
              std::vector<T>& x,
              std::vector<T>& y,
              std::vector<T>& z,
              std::vector<T>& h,
              Vectors&... particleProperties)
    {
        initBounds(x.size());
        checkSizesEqual(x.size(), particleKeys, x, y, z, h, particleProperties...);

        /* Global tree build and assignment ******************************************************/

        LocalIndex newNParticlesAssigned = global_.assign(
            bufDesc_.start, bufDesc_.end, reorderFunctor, particleKeys.data(), x.data(), y.data(), z.data());

        /* Domain particles update phase *********************************************************/

        size_t exchangeSize = std::max(x.size(), size_t(newNParticlesAssigned));
        reallocate(exchangeSize, particleKeys, x, y, z, h, particleProperties...);

        auto [exchangeStart, keyView] = global_.distribute(
            bufDesc_.start, bufDesc_.end, x.size(), reorderFunctor, particleKeys.data(), x.data(),
            y.data(), z.data(), h.data(), particleProperties.data()...);

        Box<T> box             = global_.box();
        std::vector<int> peers = global_.findPeers(theta_);

        /* Focused tree build ********************************************************************/

        if (firstCall_)
        {
            focusTree_.converge(box, keyView, peers, global_.assignment(), global_.tree(), global_.nodeCounts());
        }

        focusTree_.updateTree(peers, global_.assignment(), global_.tree());
        focusTree_.updateCriteria(box, keyView, peers, global_.assignment(), global_.tree(), global_.nodeCounts());

        /* Halo discovery ***********************************************************************/

        halos_.discover(focusTree_.octree(), focusTree_.assignment(), keyView, box, h.data());

        reallocate(nNodes(focusTree_.treeLeaves()) + 1, layout_);
        halos_.computeLayout(focusTree_.treeLeaves(), focusTree_.leafCounts(), focusTree_.assignment(), keyView, peers,
                             layout_);

        auto newParticleStart = layout_[focusTree_.assignment()[myRank_].start()];
        auto numParticles     = layout_.back();

        /* Rearrange particle buffers ************************************************************/

        reallocate(numParticles, x, y, z, h, particleProperties..., swapSpace_, swapKeys_);
        reorderArrays(reorderFunctor, exchangeStart, newParticleStart, x.data(), y.data(), z.data(), /* no h */
                      particleProperties.data()...);

        omp_copy(h.begin(), h.begin() + keyView.size(), swapSpace_.begin() + newParticleStart);
        swap(h, swapSpace_);
        omp_copy(keyView.begin(), keyView.end(), swapKeys_.begin() + newParticleStart);
        swap(particleKeys, swapKeys_);

        bufDesc_.start = newParticleStart;
        bufDesc_.end   = layout_[focusTree_.assignment()[myRank_].end()];

        /* Halo exchange *************************************************************************/

        exchangeHalos(x, y, z, h);

        // compute SFC keys of received halo particles
        computeSfcKeys(x.data(), y.data(), z.data(), sfcKindPointer(particleKeys.data()), bufDesc_.start, box);
        computeSfcKeys(x.data() + bufDesc_.end, y.data() + bufDesc_.end, z.data() + bufDesc_.end,
                       sfcKindPointer(particleKeys.data()) + bufDesc_.end, x.size() - bufDesc_.end, box);

        firstCall_ = false;
    }

    //! @brief repeat the halo exchange pattern from the previous sync operation for a different set of arrays
    template<class... Arrays>
    void exchangeHalos(Arrays&... arrays) const
    {
        halos_.exchangeHalos(arrays...);
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
    T addGravityAcceleration(gsl::span<const T> x,
                             gsl::span<const T> y,
                             gsl::span<const T> z,
                             gsl::span<const T> h,
                             gsl::span<const T> m,
                             float G,
                             gsl::span<T> ax,
                             gsl::span<T> ay,
                             gsl::span<T> az)
    {
        const Octree<KeyType>& octree = focusTree_.octree();
        std::vector<GravityMultipole<T>> multipoles(octree.numTreeNodes());
        computeMultipoles(octree, layout_, x.data(), y.data(), z.data(), m.data(), multipoles.data());

        return computeGravity(octree, multipoles.data(), layout_.data(), 0, octree.numLeafNodes(), x.data(), y.data(),
                              z.data(), h.data(), m.data(), global_.box(), theta_, G, ax.data(), ay.data(),
                              az.data());
    }

    //! @brief return the index of the first particle that's part of the local assignment
    [[nodiscard]] LocalIndex startIndex() const { return bufDesc_.start; }

    //! @brief return one past the index of the last particle that's part of the local assignment
    [[nodiscard]] LocalIndex endIndex() const { return bufDesc_.end; }

    //! @brief return number of locally assigned particles
    [[nodiscard]] LocalIndex nParticles() const { return endIndex() - startIndex(); }

    //! @brief return number of locally assigned particles plus number of halos
    [[nodiscard]] LocalIndex nParticlesWithHalos() const { return layout_.back(); }

    //! @brief read only visibility of the global octree leaves to the outside
    gsl::span<const KeyType> tree() const { return global_.tree(); }

    //! @brief read only visibility of the focused octree leaves to the outside
    gsl::span<const KeyType> focusTree() const { return focusTree_.treeLeaves(); }

    //! @brief return the coordinate bounding box from the previous sync call
    Box<T> box() const { return global_.box(); }

private:

    //! @brief bounds initialization on first call, use all particles
    template<class... Arrays>
    void initBounds(std::size_t bufferSize)
    {
        if (firstCall_)
        {
            bufDesc_ = {0, LocalIndex(bufferSize), LocalIndex(bufferSize)};

            //particleStart_ = 0;
            //particleEnd_   = bufferSize;
            layout_        = {0, LocalIndex(bufferSize)};
        }
    }

    //! @brief make sure all array sizes are equal to @p value
    template<class... Arrays>
    static void checkSizesEqual(std::size_t value, const Arrays&... arrays)
    {
        std::array<std::size_t, sizeof...(Arrays)> sizes{arrays.size()...};
        bool allEqual = size_t(std::count(begin(sizes), end(sizes), value)) == sizes.size();
        if (!allEqual)
        {
            throw std::runtime_error("Domain sync: input array sizes are inconsistent\n");
        }
    }

    int myRank_;
    int numRanks_;
    unsigned bucketSizeFocus_;

    //! @brief MAC parameter for focus resolution and gravity treewalk
    float theta_;

    /*! @brief array index of first local particle belonging to the assignment
     *  i.e. the index of the first particle that belongs to this rank and is not a halo.
     */
    //LocalIndex particleStart_{0};
    //! @brief index (upper bound) of last particle that belongs to the assignment
    //LocalIndex particleEnd_{0};
    BufferDescription bufDesc_{0, 0, 0};

    /*! @brief locally focused, fully traversable octree, used for halo discovery and exchange
     *
     * -Uses bucketSizeFocus_ as the maximum particle count per leaf within the focused SFC area.
     * -Outside the focus area, each leaf node with a particle count larger than bucketSizeFocus_
     *  fulfills a MAC with theta as the opening parameter
     * -Also contains particle counts.
     */
    FocusedOctree<KeyType> focusTree_;

    GlobalAssignment<KeyType, T> global_;

    //! @brief particle offsets of each leaf node in focusedTree_, length = focusedTree_.treeLeaves().size()
    std::vector<LocalIndex> layout_;

    Halos<KeyType> halos_{myRank_};

    bool firstCall_{true};

    ReorderFunctor reorderFunctor;
    std::vector<T> swapSpace_;
    std::vector<KeyType> swapKeys_;
};

} // namespace cstone
