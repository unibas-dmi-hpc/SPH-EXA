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
 * @brief  Build a tree for Ryoanji with the cornerstone framework
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <vector>

#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/cuda/thrust_util.cuh"
#include "cstone/sfc/sfc_gpu.h"
#include "cstone/tree/octree_gpu.h"
#include "cstone/tree/update_gpu.cuh"

#include "ryoanji/nbody/types.h"
#include "ryoanji/interface/treebuilder.cuh"

namespace ryoanji
{

template<class KeyType>
class TreeBuilder<KeyType>::Impl
{
public:
    Impl();

    Impl(unsigned ncrit);

    template<class T>
    cstone::TreeNodeIndex update(T* x, T* y, T* z, size_t numBodies, const cstone::Box<T>& box);

    int extract(int2* h_levelRange);

    const LocalIndex* layout() const { return rawPtr(d_layout_); }

    const TreeNodeIndex* childOffsets() const { return rawPtr(octreeGpuData_.childOffsets); }

    const TreeNodeIndex* leafToInternal() const
    {
        return rawPtr(octreeGpuData_.leafToInternal) + octreeGpuData_.numInternalNodes;
    }

    const TreeNodeIndex* internalToLeaf() const { return rawPtr(octreeGpuData_.internalToLeaf); }

    TreeNodeIndex numLeafNodes() const { return octreeGpuData_.numLeafNodes; }

private:
    unsigned bucketSize_;

    thrust::device_vector<KeyType>  d_tree_;
    thrust::device_vector<unsigned> d_counts_;

    thrust::device_vector<KeyType>               tmpTree_;
    thrust::device_vector<cstone::TreeNodeIndex> workArray_;

    cstone::OctreeData<KeyType, cstone::GpuTag> octreeGpuData_;
    thrust::device_vector<cstone::LocalIndex>   d_layout_;
};

template<class KeyType>
TreeBuilder<KeyType>::Impl::Impl()
    : bucketSize_(64)
{
}

template<class KeyType>
TreeBuilder<KeyType>::Impl::Impl(unsigned ncrit)
    : bucketSize_(ncrit)
{
}

template<class KeyType>
template<class T>
cstone::TreeNodeIndex TreeBuilder<KeyType>::Impl::update(T* x, T* y, T* z, size_t numBodies,
                                                         const cstone::Box<T>& csBox)
{
    thrust::device_vector<KeyType> d_keys(numBodies);
    thrust::device_vector<int>     d_ordering(numBodies);
    thrust::device_vector<T>       tmp(numBodies);

    cstone::computeSfcKeysGpu(x, y, z, cstone::sfcKindPointer(rawPtr(d_keys)), numBodies, csBox);

    thrust::sequence(d_ordering.begin(), d_ordering.end(), 0);
    thrust::sort_by_key(thrust::device, d_keys.begin(), d_keys.end(), d_ordering.begin());

    thrust::gather(thrust::device, d_ordering.begin(), d_ordering.end(), x, tmp.begin());
    thrust::copy(tmp.begin(), tmp.end(), x);
    thrust::gather(thrust::device, d_ordering.begin(), d_ordering.end(), y, tmp.begin());
    thrust::copy(tmp.begin(), tmp.end(), y);
    thrust::gather(thrust::device, d_ordering.begin(), d_ordering.end(), z, tmp.begin());
    thrust::copy(tmp.begin(), tmp.end(), z);

    if (d_tree_.size() == 0)
    {
        // initial guess on first call. use previous tree as guess on subsequent calls
        d_tree_   = std::vector<KeyType>{0, cstone::nodeRange<KeyType>(0)};
        d_counts_ = std::vector<unsigned>{unsigned(numBodies)};
    }

    while (!cstone::updateOctreeGpu(rawPtr(d_keys), rawPtr(d_keys) + d_keys.size(), bucketSize_, d_tree_, d_counts_,
                                    tmpTree_, workArray_))
        ;

    octreeGpuData_.resize(cstone::nNodes(d_tree_));
    cstone::buildOctreeGpu(rawPtr(d_tree_), octreeGpuData_.data());

    return octreeGpuData_.numInternalNodes + octreeGpuData_.numLeafNodes;
}

template<class KeyType>
int TreeBuilder<KeyType>::Impl::extract(int2* h_levelRange)
{
    d_layout_.resize(d_counts_.size() + 1);
    thrust::copy(d_counts_.begin(), d_counts_.end(), d_layout_.begin());
    thrust::exclusive_scan(d_layout_.data(), d_layout_.data() + d_layout_.size(), d_layout_.data());

    std::vector<int> cs_levelRange = toHost(octreeGpuData_.levelRange);

    int numLevels = 0;
    for (int level = 0; level <= cstone::maxTreeLevel<KeyType>{}; ++level)
    {
        if (cs_levelRange[level] == cs_levelRange[level + 1])
        {
            numLevels = level - 1;
            break;
        }

        h_levelRange[level].x = cs_levelRange[level];
        h_levelRange[level].y = cs_levelRange[level + 1];
    }

    return numLevels;
}

template<class KeyType>
TreeBuilder<KeyType>::TreeBuilder()
    : impl_(new Impl())
{
}

template<class KeyType>
TreeBuilder<KeyType>::TreeBuilder(unsigned ncrit)
    : impl_(new Impl(ncrit))
{
}

template<class KeyType>
TreeBuilder<KeyType>::~TreeBuilder() = default;

template<class KeyType>
template<class T>
int TreeBuilder<KeyType>::update(T* x, T* y, T* z, size_t numBodies, const cstone::Box<T>& box)
{
    return impl_->update(x, y, z, numBodies, box);
}

template<class KeyType>
int TreeBuilder<KeyType>::extract(int2* h_levelRange)
{
    return impl_->extract(h_levelRange);
}

template<class KeyType>
const LocalIndex* TreeBuilder<KeyType>::layout() const
{
    return impl_->layout();
}

template<class KeyType>
const TreeNodeIndex* TreeBuilder<KeyType>::childOffsets() const
{
    return impl_->childOffsets();
}

template<class KeyType>
const TreeNodeIndex* TreeBuilder<KeyType>::leafToInternal() const
{
    return impl_->leafToInternal();
}

template<class KeyType>
const TreeNodeIndex* TreeBuilder<KeyType>::internalToLeaf() const
{
    return impl_->internalToLeaf();
}

template<class KeyType>
TreeNodeIndex TreeBuilder<KeyType>::numLeafNodes() const
{
    return impl_->numLeafNodes();
}

template<class KeyType>
unsigned TreeBuilder<KeyType>::maxTreeLevel() const
{
    return cstone::maxTreeLevel<KeyType>{};
}

template class TreeBuilder<uint32_t>;
template class TreeBuilder<uint64_t>;

template int TreeBuilder<uint32_t>::update(float*, float*, float*, size_t, const cstone::Box<float>&);
template int TreeBuilder<uint64_t>::update(float*, float*, float*, size_t, const cstone::Box<float>&);
template int TreeBuilder<uint32_t>::update(double*, double*, double*, size_t, const cstone::Box<double>&);
template int TreeBuilder<uint64_t>::update(double*, double*, double*, size_t, const cstone::Box<double>&);

} // namespace ryoanji
