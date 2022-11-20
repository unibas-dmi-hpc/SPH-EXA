/*
 * MIT License
 *
 * Copyright (c) 2022 CSCS, ETH Zurich
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

#include "cstone/cuda/cuda_utils.hpp"
#include "cstone/domain/assignment.hpp"
#ifdef USE_CUDA
#include "cstone/domain/assignment_gpu.cuh"
#include "cstone/primitives/gather.cuh"
#endif
#include "cstone/domain/exchange_keys.hpp"
#include "cstone/domain/layout.hpp"
#include "cstone/focus/octree_focus_mpi.hpp"
#include "cstone/halos/halos.hpp"
#include "cstone/primitives/gather.hpp"
#include "cstone/traversal/collisions.hpp"
#include "cstone/traversal/peers.hpp"
#include "cstone/tree/accel_switch.hpp"
#include "cstone/sfc/box_mpi.hpp"
#include "cstone/sfc/sfc.hpp"
#include "cstone/sfc/sfc_gpu.h"
#include "cstone/util/reallocate.hpp"
#include "cstone/util/traits.hpp"

namespace cstone
{

template<class KeyType, class T>
class GlobalAssignmentGpu;

template<class IndexType, class BufferType>
class GpuSfcSorter;

template<class KeyType, class T, class Accelerator = CpuTag>
class Domain
{
    static_assert(std::is_unsigned<KeyType>{}, "SFC key type needs to be an unsigned integer\n");

    //! @brief A vector template that resides on the hardware specified as Accelerator
    template<class ValueType>
    using AccVector =
        typename AccelSwitchType<Accelerator, std::vector, thrust::device_vector>::template type<ValueType>;

    template<class BufferType>
    using ReorderFunctor_t =
        typename AccelSwitchType<Accelerator, SfcSorter, GpuSfcSorter>::template type<LocalIndex, BufferType>;

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
    template<class KeyVec, class VectorX, class VectorH, class... Vectors1, class... Vectors2>
    void sync(KeyVec& particleKeys,
              VectorX& x,
              VectorX& y,
              VectorX& z,
              VectorH& h,
              std::tuple<Vectors1&...> particleProperties,
              std::tuple<Vectors2&...> scratchBuffers)
    {
        staticChecks<KeyVec, VectorX, VectorH, Vectors1...>(scratchBuffers);
        auto& sfcOrder = std::get<sizeof...(Vectors2) - 1>(scratchBuffers);
        ReorderFunctor_t<std::decay_t<decltype(sfcOrder)>> reorderer(sfcOrder);

        auto scratch = discardLastElement(scratchBuffers);

        auto [exchangeStart, keyView] =
            distribute(reorderer, particleKeys, x, y, z, std::tuple_cat(std::tie(h), particleProperties), scratch);
        // h is already reordered here for use in halo discovery
        reorderArrays(reorderer, exchangeStart, 0, std::tie(h), scratch);

        float invThetaEff      = invThetaMinMac(theta_);
        std::vector<int> peers = findPeersMac(myRank_, global_.assignment(), global_.octree(), box(), invThetaEff);

        if (firstCall_)
        {
            focusTree_.converge(box(), keyView, peers, global_.assignment(), global_.treeLeaves(), global_.nodeCounts(),
                                invThetaEff, std::get<0>(scratch));
        }
        focusTree_.updateTree(peers, global_.assignment(), global_.treeLeaves());
        focusTree_.updateCounts(keyView, global_.treeLeaves(), global_.nodeCounts(), std::get<0>(scratch));
        focusTree_.updateMinMac(box(), global_.assignment(), global_.treeLeaves(), invThetaEff);

        uploadOctree();

        auto octreeView            = focusTree_.octree().data();
        const KeyType* focusLeaves = focusTree_.treeLeaves().data();
        if constexpr (HaveGpu<Accelerator>{})
        {
            octreeView  = ((const decltype(octreeAcc_)&)octreeAcc_).data();
            focusLeaves = rawPtr(focusLeavesAcc_);
        }

        reallocate(layout_, octreeView.numLeafNodes + 1, 1.01);
        halos_.discover(octreeView.prefixes, octreeView.childOffsets, octreeView.internalToLeaf, focusLeaves,
                        focusTree_.leafCounts(), focusTree_.assignment(), layout_, box(), rawPtr(h),
                        std::get<0>(scratch));
        halos_.computeLayout(focusTree_.treeLeaves(), focusTree_.leafCounts(), focusTree_.assignment(), peers, layout_);

        updateLayout(reorderer, exchangeStart, keyView, particleKeys, std::tie(h),
                     std::tuple_cat(std::tie(x, y, z), particleProperties), scratch);
        setupHalos(particleKeys, x, y, z, h, scratch);
        firstCall_ = false;
    }

    template<class KeyVec, class VectorX, class VectorH, class VectorM, class... Vectors1, class... Vectors2>
    void syncGrav(KeyVec& particleKeys,
                  VectorX& x,
                  VectorX& y,
                  VectorX& z,
                  VectorH& h,
                  VectorM& m,
                  std::tuple<Vectors1&...> particleProperties,
                  std::tuple<Vectors2&...> scratchBuffers)
    {
        staticChecks<KeyVec, VectorX, VectorH, VectorM, Vectors1...>(scratchBuffers);
        auto& sfcOrder = std::get<sizeof...(Vectors2) - 1>(scratchBuffers);
        ReorderFunctor_t<std::decay_t<decltype(sfcOrder)>> reorderer(sfcOrder);

        auto scratch = discardLastElement(scratchBuffers);

        auto [exchangeStart, keyView] =
            distribute(reorderer, particleKeys, x, y, z, std::tuple_cat(std::tie(h, m), particleProperties), scratch);
        reorderArrays(reorderer, exchangeStart, 0, std::tie(x, y, z, h, m), scratch);

        float invThetaEff      = invThetaVecMac(theta_);
        std::vector<int> peers = findPeersMac(myRank_, global_.assignment(), global_.octree(), box(), invThetaEff);

        if (firstCall_)
        {
            int converged = 0;
            while (converged != numRanks_)
            {
                focusTree_.updateMinMac(box(), global_.assignment(), global_.treeLeaves(), invThetaEff);
                converged = focusTree_.updateTree(peers, global_.assignment(), global_.treeLeaves());
                focusTree_.updateCounts(keyView, global_.treeLeaves(), global_.nodeCounts(), std::get<0>(scratch));
                focusTree_.updateCenters(rawPtr(x), rawPtr(y), rawPtr(z), rawPtr(m), global_.assignment(),
                                         global_.octree(), box(), std::get<0>(scratch), std::get<1>(scratch));
                focusTree_.updateMacs(box(), global_.assignment(), global_.treeLeaves());
                MPI_Allreduce(MPI_IN_PLACE, &converged, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            }
        }
        focusTree_.updateMinMac(box(), global_.assignment(), global_.treeLeaves(), invThetaEff);
        focusTree_.updateTree(peers, global_.assignment(), global_.treeLeaves());
        focusTree_.updateCounts(keyView, global_.treeLeaves(), global_.nodeCounts(), std::get<0>(scratch));
        focusTree_.updateCenters(rawPtr(x), rawPtr(y), rawPtr(z), rawPtr(m), global_.assignment(), global_.octree(),
                                 box(), std::get<0>(scratch), std::get<1>(scratch));
        focusTree_.updateMacs(box(), global_.assignment(), global_.treeLeaves());

        uploadOctree();

        auto octreeView            = focusTree_.octree().data();
        const KeyType* focusLeaves = focusTree_.treeLeaves().data();
        if constexpr (HaveGpu<Accelerator>{})
        {
            octreeView  = ((const decltype(octreeAcc_)&)octreeAcc_).data();
            focusLeaves = rawPtr(focusLeavesAcc_);
        }

        reallocate(layout_, octreeView.numLeafNodes + 1, 1.01);
        halos_.discover(octreeView.prefixes, octreeView.childOffsets, octreeView.internalToLeaf, focusLeaves,
                        focusTree_.leafCounts(), focusTree_.assignment(), layout_, box(), rawPtr(h),
                        std::get<0>(scratch));
        focusTree_.addMacs(halos_.haloFlags());
        halos_.computeLayout(focusTree_.treeLeaves(), focusTree_.leafCounts(), focusTree_.assignment(), peers, layout_);

        // diagnostics(keyView.size(), peers);

        updateLayout(reorderer, exchangeStart, keyView, particleKeys, std::tie(x, y, z, h, m), particleProperties,
                     scratch);
        setupHalos(particleKeys, x, y, z, h, scratch);
        firstCall_ = false;
    }

    //! @brief repeat the halo exchange pattern from the previous sync operation for a different set of arrays
    template<class... Vectors, class SendBuffer, class ReceiveBuffer>
    void exchangeHalos(std::tuple<Vectors&...> arrays, SendBuffer& sendBuffer, ReceiveBuffer& receiveBuffer) const
    {
        std::apply([this](auto&... arrays) { this->template checkSizesEqual(this->bufDesc_.size, arrays...); }, arrays);
        this->halos_.exchangeHalos(arrays, sendBuffer, receiveBuffer);
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
    const Octree<KeyType>& globalTree() const { return global_.octree(); }
    //! @brief read only visibility of the focused octree
    const FocusedOctree<KeyType, T, Accelerator>& focusTree() const { return focusTree_; }
    //! @brief the index of the first locally assigned cell in focusTree()
    TreeNodeIndex startCell() const { return focusTree_.assignment()[myRank_].start(); }
    //! @brief the index of the last locally assigned cell in focusTree()
    TreeNodeIndex endCell() const { return focusTree_.assignment()[myRank_].end(); }
    //! @brief particle offsets of each focus tree leaf cell
    gsl::span<const LocalIndex> layout() const { return layout_; }
    //! @brief return the coordinate bounding box from the previous sync call
    const Box<T>& box() const { return global_.box(); }

private:
    //! @brief bounds initialization on first call, use all particles
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
        if (!allEqual) { throw std::runtime_error("Domain sync: input array sizes are inconsistent\n"); }
    }

    /*! @brief check type requirements on scratch buffers
     *
     * @tparam KeyVec           type of vector used to store SFC keys
     * @tparam ConservedVectors types of conserved particle field vectors (x,y,z,...)
     * @param  scratchBuffers   a tuple of references to vectors for scratch usage
     *
     * At least 3 scratch buffers are needed. 2 for send/receive and the last one is used to store the SFC ordering.
     * An additional requirement is that for each value type (float, double) appearing in the list of conserved
     * vectors, a scratch buffers with a matching type is needed to allow for vector swaps.
     */
    template<class KeyVec, class... ConservedVectors, class ScratchBuffers>
    void staticChecks(ScratchBuffers& scratchBuffers)
    {
        static_assert(std::is_same_v<typename KeyVec::value_type, KeyType>);
        static_assert(std::tuple_size_v<ScratchBuffers> >= 3);

        auto tup               = discardLastElement(scratchBuffers);
        constexpr auto indices = std::make_tuple(util::FindIndex<ConservedVectors&, std::decay_t<decltype(tup)>>{}...);

        auto valueTypeCheck = [](auto index)
        {
            static_assert(
                index < std::tuple_size_v<std::decay_t<decltype(tup)>>,
                "one of the conserved fields has a value type that was not found among the available scratch buffers");
        };
        for_each_tuple(valueTypeCheck, indices);
    }

    template<class Reord, class KeyVec, class VectorX, class... Vectors1, class... Vectors2>
    auto distribute(Reord& reorderFunctor,
                    KeyVec& keys,
                    VectorX& x,
                    VectorX& y,
                    VectorX& z,
                    std::tuple<Vectors1&...> particleProperties,
                    std::tuple<Vectors2&...> scratchBuffers)
    {
        initBounds(x.size());
        auto distributedArrays = std::tuple_cat(std::tie(keys, x, y, z), particleProperties);
        std::apply([size = x.size()](auto&... arrays) { checkSizesEqual(size, arrays...); }, distributedArrays);

        // Global tree build and assignment
        LocalIndex newNParticlesAssigned =
            global_.assign(bufDesc_, reorderFunctor, rawPtr(keys), rawPtr(x), rawPtr(y), rawPtr(z));

        size_t exchangeSize = std::max(x.size(), size_t(newNParticlesAssigned));
        lowMemReallocate(exchangeSize, 1.01, distributedArrays, scratchBuffers);

        return std::apply(
            [&reorderFunctor, &scratchBuffers, this](auto&... arrays)
            {
                return global_.distribute(bufDesc_, reorderFunctor, std::get<0>(scratchBuffers),
                                          std::get<1>(scratchBuffers), rawPtr(arrays)...);
            },
            distributedArrays);
    }

    template<class KeyVec, class VectorX, class VectorH, class... Vs>
    void setupHalos(KeyVec& keys, VectorX& x, VectorX& y, VectorX& z, VectorH& h, std::tuple<Vs&...> scratch)
    {
        exchangeHalos(std::tie(x, y, z, h), std::get<0>(scratch), std::get<1>(scratch));

        // compute SFC keys of received halo particles
        if constexpr (IsDeviceVector<KeyVec>{})
        {
            computeSfcKeysGpu(rawPtr(x), rawPtr(y), rawPtr(z), sfcKindPointer(rawPtr(keys)), bufDesc_.start, box());
            computeSfcKeysGpu(rawPtr(x) + bufDesc_.end, rawPtr(y) + bufDesc_.end, rawPtr(z) + bufDesc_.end,
                              sfcKindPointer(rawPtr(keys)) + bufDesc_.end, x.size() - bufDesc_.end, box());
        }
        else
        {
            computeSfcKeys(rawPtr(x), rawPtr(y), rawPtr(z), sfcKindPointer(rawPtr(keys)), bufDesc_.start, box());
            computeSfcKeys(rawPtr(x) + bufDesc_.end, rawPtr(y) + bufDesc_.end, rawPtr(z) + bufDesc_.end,
                           sfcKindPointer(rawPtr(keys)) + bufDesc_.end, x.size() - bufDesc_.end, box());
        }
    }

    void uploadOctree()
    {
        if constexpr (HaveGpu<Accelerator>{})
        {
            auto& octree               = focusTree_.octree();
            TreeNodeIndex numLeafNodes = octree.numLeafNodes();
            TreeNodeIndex numNodes     = octree.numTreeNodes();

            octreeAcc_.resize(numLeafNodes);
            reallocateDestructive(focusLeavesAcc_, numLeafNodes + 1, 1.01);

            memcpyH2D(octree.nodeKeys().data(), numNodes, rawPtr(octreeAcc_.prefixes));
            memcpyH2D(octree.childOffsets().data(), numNodes, rawPtr(octreeAcc_.childOffsets));
            memcpyH2D(octree.parents().data(), octree.parents().size(), rawPtr(octreeAcc_.parents));
            memcpyH2D(octree.levelRange().data(), octree.levelRange().size(), rawPtr(octreeAcc_.levelRange));
            memcpyH2D(octree.toLeafOrder().data(), numNodes, rawPtr(octreeAcc_.internalToLeaf));
            memcpyH2D(octree.internalOrder().data(), numLeafNodes,
                      rawPtr(octreeAcc_.leafToInternal) + octree.numInternalNodes());

            const auto& leaves = focusTree_.treeLeaves().data();
            memcpyH2D(leaves, numLeafNodes + 1, rawPtr(focusLeavesAcc_));
        }
    }

    template<class Reord, class KeyVec, class... Arrays1, class... Arrays2, class... Arrays3>
    void updateLayout(Reord& reorderFunctor,
                      LocalIndex exchangeStart,
                      gsl::span<const KeyType> keyView,
                      KeyVec& keys,
                      std::tuple<Arrays1&...> orderedBuffers,
                      std::tuple<Arrays2&...> unorderedBuffers,
                      std::tuple<Arrays3&...> scratchBuffers)
    {
        auto myRange = focusTree_.assignment()[myRank_];
        BufferDescription newBufDesc{layout_[myRange.start()], layout_[myRange.end()], layout_.back()};

        lowMemReallocate(newBufDesc.size, 1.01, std::tuple_cat(orderedBuffers, unorderedBuffers), scratchBuffers);

        // re-locate particle SFC keys
        if constexpr (IsDeviceVector<KeyVec>{})
        {
            auto& swapSpace = std::get<0>(scratchBuffers);
            size_t origSize = reallocateBytes(swapSpace, keyView.size() * sizeof(KeyType));

            auto* swapPtr = reinterpret_cast<KeyType*>(rawPtr(swapSpace));
            memcpyD2D(keyView.data(), keyView.size(), swapPtr);
            reallocateDestructive(keys, newBufDesc.size, 1.01);
            memcpyD2D(swapPtr, keyView.size(), rawPtr(keys) + newBufDesc.start);

            reallocate(swapSpace, origSize, 1.0);
        }
        else
        {
            reallocate(swapKeys_, newBufDesc.size, 1.01);
            omp_copy(keyView.begin(), keyView.end(), swapKeys_.begin() + newBufDesc.start);
            swap(keys, swapKeys_);
        }

        // relocate ordered buffer contents from offset 0 to offset newBufDesc.start
        auto relocate = [size = keyView.size(), dest = newBufDesc.start, &scratchBuffers](auto& array)
        {
            static_assert(util::FindIndex<decltype(array), std::tuple<Arrays3&...>>{} < sizeof...(Arrays3),
                          "No suitable scratch buffer available");
            auto& swapSpace = util::pickType<decltype(array)>(scratchBuffers);
            if constexpr (IsDeviceVector<std::decay_t<decltype(array)>>{})
            {
                memcpyD2D(rawPtr(array), size, rawPtr(swapSpace) + dest);
            }
            else { omp_copy(array.begin(), array.begin() + size, swapSpace.begin() + dest); }
            swap(array, swapSpace);
        };
        for_each_tuple(relocate, orderedBuffers);

        // reorder the unordered buffers
        reorderArrays(reorderFunctor, exchangeStart, newBufDesc.start, unorderedBuffers, scratchBuffers);

        // newBufDesc is now the valid buffer description
        std::swap(newBufDesc, bufDesc_);
    }

    void diagnostics(size_t assignedSize, gsl::span<int> peers)
    {
        auto focusAssignment = focusTree_.assignment();
        auto focusTree       = focusTree_.treeLeaves();
        auto globalTree      = global_.treeLeaves();

        TreeNodeIndex numFocusPeers    = 0;
        TreeNodeIndex numFocusTruePeer = 0;
        for (int i = 0; i < numRanks_; ++i)
        {
            if (i != myRank_)
            {
                numFocusPeers += focusAssignment[i].count();
                for (TreeNodeIndex fi = focusAssignment[i].start(); fi < focusAssignment[i].end(); ++fi)
                {
                    KeyType fnstart  = focusTree[fi];
                    KeyType fnend    = focusTree[fi + 1];
                    TreeNodeIndex gi = findNodeAbove(globalTree, fnstart);
                    if (!(gi < nNodes(globalTree) && globalTree[gi] == fnstart && globalTree[gi + 1] <= fnend))
                    {
                        numFocusTruePeer++;
                    }
                }
            }
        }

        int numFlags = std::count(halos_.haloFlags().cbegin(), halos_.haloFlags().cend(), 1);
        for (int i = 0; i < numRanks_; ++i)
        {
            if (i == myRank_)
            {
                std::cout << "rank " << i << " " << assignedSize << " " << layout_.back()
                          << " focus h/true/peers/loc/tot: " << numFlags << "/" << numFocusTruePeer << "/"
                          << numFocusPeers << "/" << focusAssignment[myRank_].count() << "/"
                          << halos_.haloFlags().size() << " peers: [" << peers.size() << "] ";
                if (numRanks_ <= 32)
                {
                    for (auto r : peers)
                    {
                        std::cout << r << " ";
                    }
                }
                std::cout << std::endl;
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
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
    FocusedOctree<KeyType, T, Accelerator> focusTree_;

    using Distributor_t =
        typename AccelSwitchType<Accelerator, GlobalAssignment, GlobalAssignmentGpu>::template type<KeyType, T>;
    Distributor_t global_;

    OctreeData<KeyType, Accelerator> octreeAcc_;

    AccVector<KeyType> focusLeavesAcc_;
    AccVector<unsigned> focusCountsAcc_;

    //! @brief particle offsets of each leaf node in focusedTree_, length = focusedTree_.treeLeaves().size()
    std::vector<LocalIndex> layout_;

    Halos<KeyType, Accelerator> halos_{myRank_};

    bool firstCall_{true};

    std::vector<KeyType> swapKeys_;
};

} // namespace cstone
