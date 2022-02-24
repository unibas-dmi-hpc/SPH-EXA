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
#include "cstone/focus/octree_focus_mpi.hpp"
#include "cstone/halos/exchange_halos.hpp"
#include "cstone/halos/halos.hpp"
#include "cstone/traversal/collisions.hpp"
#include "cstone/traversal/peers.hpp"
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
        , focusTree_(rank, numRanks_, bucketSizeFocus_, theta_)
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
     * @param[out]   particleKeys        SFC particleKeys
     * @param[inout] x                   floating point coordinates
     * @param[inout] y
     * @param[inout] z
     * @param[inout] h                   interaction radii in SPH convention, actual interaction radius
     *                                   is twice the value in h
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
        auto [exchangeStart, keyView] = distribute(particleKeys, x, y, z, h, particleProperties...);
        // h is already reordered here for use in halo discovery
        reorderFunctor(h.data() + exchangeStart, h.data());

        std::vector<int> peers = findPeersMac(myRank_, global_.assignment(), global_.octree(), box(), theta_);

        if (firstCall_)
        {
            focusTree_.converge(box(), keyView, peers, global_.assignment(), global_.treeLeaves(), global_.nodeCounts());
        }
        focusTree_.updateTree(peers, global_.assignment(), global_.treeLeaves());
        focusTree_.updateCounts(keyView, peers, global_.treeLeaves(), global_.nodeCounts());
        focusTree_.updateMinMac(box(), global_.assignment(), global_.treeLeaves());

        halos_.discover(focusTree_.octree(), focusTree_.assignment(), keyView, box(), h.data());

        reallocate(nNodes(focusTree_.treeLeaves()) + 1, layout_);
        halos_.computeLayout(focusTree_.treeLeaves(), focusTree_.leafCounts(), focusTree_.assignment(), keyView, peers,
                             layout_);

        updateLayout(exchangeStart, keyView, particleKeys, std::tie(h), std::tie(x, y, z, particleProperties...));
        setupHalos(particleKeys, x, y, z, h);
        firstCall_ = false;
    }

    template<class... Vectors>
    void syncGrav(std::vector<KeyType>& particleKeys,
                  std::vector<T>& x,
                  std::vector<T>& y,
                  std::vector<T>& z,
                  std::vector<T>& h,
                  std::vector<T>& m,
                  Vectors&... particleProperties)
    {
        auto [exchangeStart, keyView] = distribute(particleKeys, x, y, z, h, m, particleProperties...);
        reorderArrays(reorderFunctor, exchangeStart, 0, x.data(), y.data(), z.data(), h.data(), m.data());

        std::vector<int> peers = findPeersMac(myRank_, global_.assignment(), global_.octree(), box(), theta_);

        if (firstCall_)
        {
            focusTree_.converge(box(), keyView, peers, global_.assignment(), global_.treeLeaves(), global_.nodeCounts());
        }
        focusTree_.updateTree(peers, global_.assignment(), global_.treeLeaves());
        focusTree_.updateCounts(keyView, peers, global_.treeLeaves(), global_.nodeCounts());
        focusTree_.template updateCenters<T, T>(x, y, z, m, peers, global_.assignment(), global_.octree(), box());
        focusTree_.updateVecMac(box(), global_.assignment(), global_.treeLeaves());

        halos_.discover(focusTree_.octree(), focusTree_.assignment(), keyView, box(), h.data());
        focusTree_.addMacs(halos_.haloFlags());

        reallocate(nNodes(focusTree_.treeLeaves()) + 1, layout_);
        halos_.computeLayout(focusTree_.treeLeaves(), focusTree_.leafCounts(), focusTree_.assignment(), keyView, peers,
                             layout_);

        updateLayout(exchangeStart, keyView, particleKeys, std::tie(x, y, z, h, m), std::tie(particleProperties...));
        setupHalos(particleKeys, x, y, z, h);
        firstCall_ = false;
    }

    //! @brief repeat the halo exchange pattern from the previous sync operation for a different set of arrays
    template<class... Arrays>
    void exchangeHalos(Arrays&... arrays) const
    {
        checkSizesEqual(bufDesc_.size, arrays...);
        halos_.exchangeHalos(arrays.data()...);
    }

    template<class CellProperty, class CombinationFunction>
    void exchangeFocusGlobal(gsl::span<CellProperty> cellProperties, CombinationFunction combinationFunction)
    {
        const Octree<KeyType>& globalTree = global_.octree();

        gsl::span<const SourceCenterType<T>> globalCenters = focusTree_.globalExpansionCenters();
        assert(globalTree.numTreeNodes() == globalCenters.ssize());
        combinationFunction.setCenters(globalCenters.data());

        std::vector<int> peers = findPeersMac(myRank_, global_.assignment(), globalTree, box(), theta_);

        std::vector<CellProperty> globalProperties(globalTree.numTreeNodes());

        focusTree_.peerExchange(peers, cellProperties, static_cast<int>(P2pTags::focusPeerCenters) + 1);
        focusTree_.globalExchange(globalTree, globalProperties.data(), cellProperties.data(), combinationFunction);
    }

    //! @brief return the index of the first particle that's part of the local assignment
    [[nodiscard]] LocalIndex startIndex() const { return bufDesc_.start; }
    //! @brief return one past the index of the last particle that's part of the local assignment
    [[nodiscard]] LocalIndex endIndex() const { return bufDesc_.end; }
    //! @brief return number of locally assigned particles
    [[nodiscard]] LocalIndex nParticles() const { return endIndex() - startIndex(); }
    //! @brief return number of locally assigned particles plus number of halos
    [[nodiscard]] LocalIndex nParticlesWithHalos() const { return bufDesc_.size; }
    //! @brief read only visibility of the global octree leaves to the outside
    gsl::span<const KeyType> tree() const { return global_.treeLeaves(); }
    //! @brief read only visibility of the focused octree
    const Octree<KeyType>& focusTree() const { return focusTree_.octree(); }
    //! @brief the index of the first locally assigned cell in focusTree()
    TreeNodeIndex startCell() const { return focusTree_.assignment()[myRank_].start(); }
    //! @brief the index of the last locally assigned cell in focusTree()
    TreeNodeIndex endCell() const { return focusTree_.assignment()[myRank_].end(); }
    //! @brief expansion (com) center and mac^2 radii of each focus tree cell
    gsl::span<const SourceCenterType<T>> expansionCenters() const { return focusTree_.expansionCenters(); }
    //! @brief particle offsets of each focus tree leaf cell
    gsl::span<const LocalIndex> layout() const { return layout_; }
    //! @brief return the coordinate bounding box from the previous sync call
    const Box<T>& box() const { return global_.box(); }

private:

    //! @brief bounds initialization on first call, use all particles
    template<class... Arrays>
    void initBounds(std::size_t bufferSize)
    {
        if (firstCall_)
        {
            bufDesc_ = {0, LocalIndex(bufferSize), LocalIndex(bufferSize)};
            layout_  = {0, LocalIndex(bufferSize)};
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

    template<class... Vectors>
    auto distribute(std::vector<KeyType>& particleKeys,
                    std::vector<T>& x,
                    std::vector<T>& y,
                    std::vector<T>& z,
                    Vectors&... particleProperties)
    {
        initBounds(x.size());
        checkSizesEqual(x.size(), particleKeys, x, y, z, particleProperties...);

        // Global tree build and assignment
        LocalIndex newNParticlesAssigned =
            global_.assign(bufDesc_, reorderFunctor, particleKeys.data(), x.data(), y.data(), z.data());

        size_t exchangeSize = std::max(x.size(), size_t(newNParticlesAssigned));
        reallocate(exchangeSize, particleKeys, x, y, z, particleProperties...);

        return global_.distribute(bufDesc_, reorderFunctor, particleKeys.data(), x.data(), y.data(), z.data(),
                                  particleProperties.data()...);
    }

    template<class KeyVec, class TVec>
    void setupHalos(KeyVec& keys, TVec& x, TVec& y, TVec& z, TVec& h)
    {
        exchangeHalos(x, y, z, h);

        // compute SFC keys of received halo particles
        computeSfcKeys(x.data(), y.data(), z.data(), sfcKindPointer(keys.data()), bufDesc_.start, box());
        computeSfcKeys(x.data() + bufDesc_.end, y.data() + bufDesc_.end, z.data() + bufDesc_.end,
                       sfcKindPointer(keys.data()) + bufDesc_.end, x.size() - bufDesc_.end, box());
    }

    template<class KeyVec, class... Arrays1, class... Arrays2>
    void updateLayout(LocalIndex exchangeStart,
                      gsl::span<const KeyType> keyView,
                      KeyVec& keys,
                      std::tuple<Arrays1&...> orderedBuffers,
                      std::tuple<Arrays2&...> unorderedBuffers)
    {
        auto myRange = focusTree_.assignment()[myRank_];
        BufferDescription newBufDesc{layout_[myRange.start()], layout_[myRange.end()], layout_.back()};

        // adjust sizes of all buffers if necessary
        std::apply([size = newBufDesc.size](auto&... arrays) { reallocate(size, arrays...); }, orderedBuffers);
        std::apply([size = newBufDesc.size](auto&... arrays) { reallocate(size, arrays...); }, unorderedBuffers);
        reallocate(newBufDesc.size, swapSpace_, swapKeys_);

        // relocate particle SFC keys
        omp_copy(keyView.begin(), keyView.end(), swapKeys_.begin() + newBufDesc.start);
        swap(keys, swapKeys_);

        // relocate ordered buffer contents from offset 0 to offset newBufDesc.start
        auto relocate = [size = keyView.size(), dest = newBufDesc.start, this](auto& array)
        {
            omp_copy(array.begin(), array.begin() + size, swapSpace_.begin() + dest);
            swap(array, swapSpace_);
        };
        for_each_tuple(relocate, orderedBuffers);

        // reorder the unordered buffers
        std::apply([src = exchangeStart, dest = newBufDesc.start, this](auto&... arrays)
                   { reorderArrays(reorderFunctor, src, dest, arrays.data()...); },
                   unorderedBuffers);

        // newBufDesc is now the valid buffer description
        std::swap(newBufDesc, bufDesc_);
    }

    int myRank_;
    int numRanks_;
    unsigned bucketSizeFocus_;

    //! @brief MAC parameter for focus resolution and gravity treewalk
    float theta_;

    /*! @brief description of particle buffers, storing start and end indices of assigned particles and total size
     *
     *  First element: array index of first local particle belonging to the assignment
     *  i.e. the index of the first particle that belongs to this rank and is not a halo
     *  Second element: index (upper bound) of last particle that belongs to the assignment
     */
    BufferDescription bufDesc_{0, 0, 0};

    /*! @brief locally focused, fully traversable octree, used for halo discovery and exchange
     *
     * -Uses bucketSizeFocus_ as the maximum particle count per leaf within the focused SFC area.
     * -Outside the focus area, each leaf node with a particle count larger than bucketSizeFocus_
     *  fulfills a MAC with theta as the opening parameter
     * -Also contains particle counts.
     */
    FocusedOctree<KeyType, T> focusTree_;

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
