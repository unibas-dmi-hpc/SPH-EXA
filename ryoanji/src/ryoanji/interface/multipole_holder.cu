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
 * @brief  Interface for calculation of multipole moments
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <thrust/device_vector.h>

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/util/reallocate.hpp"
#include "ryoanji/nbody/cartesian_qpole.hpp"
#include "ryoanji/nbody/upwardpass.cuh"
#include "ryoanji/nbody/upsweep_cpu.hpp"
#include "ryoanji/nbody/traversal.cuh"
#include "multipole_holder.cuh"

namespace ryoanji
{

template<class Tc, class Th, class Tm, class Ta, class Tf, class KeyType, class MType>
class MultipoleHolder<Tc, Th, Tm, Ta, Tf, KeyType, MType>::Impl
{
public:
    Impl() {}

    void upsweep(const Tc* x, const Tc* y, const Tc* z, const Tm* m, const cstone::Octree<KeyType>& globalOctree,
                 const cstone::FocusedOctree<KeyType, Tf, cstone::GpuTag>& focusTree, const cstone::LocalIndex* layout,
                 MType* multipoles)
    {
        constexpr int                  numThreads = UpsweepConfig::numThreads;
        const cstone::Octree<KeyType>& octree     = focusTree.octree();

        TreeNodeIndex numLeaves = focusTree.octree().numLeafNodes();
        resize(numLeaves);

        auto centers       = focusTree.expansionCenters();
        auto globalCenters = focusTree.globalExpansionCenters();

        const TreeNodeIndex* leafToInternal = octree.internalOrder().data();
        memcpyH2D(leafToInternal, numLeaves, rawPtr(leafToInternal_));

        const TreeNodeIndex* internalToLeaf = octree.toLeafOrder().data();
        memcpyH2D(internalToLeaf, internalToLeaf_.size(), rawPtr(internalToLeaf_));

        const TreeNodeIndex* childOffsets = octree.childOffsets().data();
        memcpyH2D(childOffsets, childOffsets_.size(), rawPtr(childOffsets_));

        memcpyH2D(layout, layout_.size(), rawPtr(layout_));
        memcpyH2D(centers.data(), centers.size(), rawPtr(centers_));

        computeLeafMultipoles<<<(numLeaves - 1) / numThreads + 1, numThreads>>>(
            x, y, z, m, rawPtr(leafToInternal_), numLeaves, rawPtr(layout_), rawPtr(centers_), rawPtr(multipoles_));

        //! first upsweep with local data
        int  numLevels  = 21;
        auto levelRange = octree.levelRange();
        for (int level = numLevels - 1; level >= 0; level--)
        {
            int numCellsLevel = levelRange[level + 1] - levelRange[level];
            int numBlocks     = (numCellsLevel - 1) / numThreads + 1;
            upsweepMultipoles<<<numBlocks, numThreads>>>(levelRange[level], levelRange[level + 1],
                                                         rawPtr(childOffsets_), rawPtr(centers_), rawPtr(multipoles_));
        }

        memcpyD2H(rawPtr(multipoles_), multipoles_.size(), multipoles);

        auto ryUpsweep = [](auto levelRange, auto childOffsets, auto M, auto centers)
        { upsweepMultipoles(levelRange, childOffsets, centers, M); };

        gsl::span multipoleSpan{multipoles, size_t(octree.numTreeNodes())};
        cstone::globalFocusExchange(globalOctree, focusTree, multipoleSpan, ryUpsweep, globalCenters.data());

        focusTree.peerExchange(multipoleSpan, static_cast<int>(cstone::P2pTags::focusPeerCenters) + 1);

        // H2D multipoles
        memcpyH2D(multipoles, multipoles_.size(), rawPtr(multipoles_));

        //! second upsweep with leaf data from peer and global ranks in place
        for (int level = numLevels - 1; level >= 0; level--)
        {
            int numCellsLevel = levelRange[level + 1] - levelRange[level];
            int numBlocks     = (numCellsLevel - 1) / numThreads + 1;
            upsweepMultipoles<<<numBlocks, numThreads>>>(levelRange[level], levelRange[level + 1],
                                                         rawPtr(childOffsets_), rawPtr(centers_), rawPtr(multipoles_));
        }
    }

    float compute(LocalIndex firstBody, LocalIndex lastBody, const Tc* x, const Tc* y, const Tc* z, const Tm* m,
                  const Th* h, Tc G, Ta* ax, Ta* ay, Ta* az)
    {
        resetTraversalCounters<<<1, 1>>>();

        constexpr int numWarpsPerBlock = TravConfig::numThreads / cstone::GpuConfig::warpSize;

        LocalIndex numBodies = lastBody - firstBody;

        // each target gets a warp (numWarps == numTargets)
        int numWarps  = (numBodies - 1) / TravConfig::targetSize + 1;
        int numBlocks = (numWarps - 1) / numWarpsPerBlock + 1;
        numBlocks     = std::min(numBlocks, TravConfig::maxNumActiveBlocks);

        LocalIndex poolSize = TravConfig::memPerWarp * numWarpsPerBlock * numBlocks;

        reallocateGeneric(globalPool_, poolSize, 1.05);
        traverse<<<numBlocks, TravConfig::numThreads>>>(
            firstBody, lastBody, {1, 9}, x, y, z, m, h, rawPtr(childOffsets_), rawPtr(internalToLeaf_), rawPtr(layout_),
            rawPtr(centers_), rawPtr(multipoles_), G, (int*)(nullptr), ax, ay, az, rawPtr(globalPool_));
        float totalPotential;
        checkGpuErrors(cudaMemcpyFromSymbol(&totalPotential, totalPotentialGlob, sizeof(float)));

        return 0.5f * Tc(G) * totalPotential;
    }

    const MType* deviceMultipoles() const { return rawPtr(multipoles_); }

private:
    void resize(size_t numLeaves)
    {
        double growthRate = 1.01;
        size_t numNodes   = numLeaves + (numLeaves - 1) / 7;

        auto dealloc = [](auto& v)
        {
            v.clear();
            v.shrink_to_fit();
        };

        if (numLeaves > leafToInternal_.capacity())
        {
            dealloc(leafToInternal_);
            dealloc(internalToLeaf_);
            dealloc(childOffsets_);
            dealloc(layout_);
            dealloc(centers_);
            dealloc(multipoles_);
        }

        reallocateGeneric(leafToInternal_, numLeaves, growthRate);
        reallocateGeneric(internalToLeaf_, numNodes, growthRate);
        reallocateGeneric(childOffsets_, numNodes, growthRate);

        reallocateGeneric(layout_, numLeaves + 1, growthRate);

        reallocateGeneric(centers_, numNodes, growthRate);
        reallocateGeneric(multipoles_, numNodes, growthRate);
    }

    thrust::device_vector<TreeNodeIndex> leafToInternal_;
    thrust::device_vector<TreeNodeIndex> internalToLeaf_;
    thrust::device_vector<TreeNodeIndex> childOffsets_;

    thrust::device_vector<LocalIndex> layout_;

    thrust::device_vector<Vec4<Tf>> centers_;
    thrust::device_vector<MType>    multipoles_;

    thrust::device_vector<int> globalPool_;
};

template<class Tc, class Th, class Tm, class Ta, class Tf, class KeyType, class MType>
MultipoleHolder<Tc, Th, Tm, Ta, Tf, KeyType, MType>::MultipoleHolder()
    : impl_(new Impl())
{
}

template<class Tc, class Th, class Tm, class Ta, class Tf, class KeyType, class MType>
MultipoleHolder<Tc, Th, Tm, Ta, Tf, KeyType, MType>::~MultipoleHolder() = default;

template<class Tc, class Th, class Tm, class Ta, class Tf, class KeyType, class MType>
void MultipoleHolder<Tc, Th, Tm, Ta, Tf, KeyType, MType>::upsweep(
    const Tc* x, const Tc* y, const Tc* z, const Tm* m, const cstone::Octree<KeyType>& globalTree,
    const cstone::FocusedOctree<KeyType, Tf, cstone::GpuTag>& focusTree, const LocalIndex* layout, MType* multipoles)
{
    impl_->upsweep(x, y, z, m, globalTree, focusTree, layout, multipoles);
}

template<class Tc, class Th, class Tm, class Ta, class Tf, class KeyType, class MType>
float MultipoleHolder<Tc, Th, Tm, Ta, Tf, KeyType, MType>::compute(LocalIndex firstBody, LocalIndex lastBody,
                                                                   const Tc* x, const Tc* y, const Tc* z, const Tm* m,
                                                                   const Th* h, Tc G, Ta* ax, Ta* ay, Ta* az)
{
    return impl_->compute(firstBody, lastBody, x, y, z, m, h, G, ax, ay, az);
}

template<class Tc, class Th, class Tm, class Ta, class Tf, class KeyType, class MType>
const MType* MultipoleHolder<Tc, Th, Tm, Ta, Tf, KeyType, MType>::deviceMultipoles() const
{
    return impl_->deviceMultipoles();
}

#define MHOLDER_SPH(Tc, Th, Tm, Ta, Tf, KeyType, MVal)                                                                 \
    template class MultipoleHolder<Tc, Th, Tm, Ta, Tf, KeyType, SphericalMultipole<MVal, 4>>

MHOLDER_SPH(double, double, double, double, double, uint64_t, double);
MHOLDER_SPH(double, double, float, double, double, uint64_t, float);
MHOLDER_SPH(float, float, float, float, float, uint64_t, float);

#define MHOLDER_CART(Tc, Th, Tm, Ta, Tf, KeyType, MVal)                                                                \
    template class MultipoleHolder<Tc, Th, Tm, Ta, Tf, KeyType, CartesianQuadrupole<MVal>>

MHOLDER_CART(double, double, double, double, double, uint64_t, double);
MHOLDER_CART(double, double, float, double, double, uint64_t, float);
MHOLDER_CART(float, float, float, float, float, uint64_t, float);

} // namespace ryoanji
