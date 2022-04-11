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
 * @brief A wrapper to select between CPU and GPU gravity implementations
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/domain/domain.hpp"
#include "ryoanji/interface/global_multipole.hpp"
#include "ryoanji/interface/multipole_holder.cuh"
#include "ryoanji/nbody/traversal_cpu.hpp"

namespace sphexa
{

template<class MType, class KeyType, class, class, class>
class MultipoleHolderCpu
{
public:
    MultipoleHolderCpu() = default;

    template<class Dataset, class Domain>
    void upsweep(const Dataset& d, const Domain& domain)
    {
        //! includes tree plus associated information, like peer ranks, assignment, counts, centers, etc
        const auto& focusTree = domain.focusTree();
        //! the focused octree, structure only
        const cstone::Octree<KeyType>& octree = focusTree.octree();

        reallocate(multipoles_, octree.numTreeNodes(), 1.05);

        ryoanji::computeGlobalMultipoles(d.x.data(),
                                         d.y.data(),
                                         d.z.data(),
                                         d.m.data(),
                                         d.x.size(),
                                         domain.globalTree(),
                                         domain.focusTree(),
                                         domain.layout().data(),
                                         multipoles_.data());
    }

    template<class Dataset, class Domain>
    void traverse(Dataset& d, const Domain& domain)
    {
        //! includes tree plus associated information, like peer ranks, assignment, counts, centers, etc
        const auto& focusTree = domain.focusTree();
        //! the focused octree, structure only
        const cstone::Octree<KeyType>& octree = focusTree.octree();

        d.egrav = ryoanji::computeGravity(octree,
                                          focusTree.expansionCenters().data(),
                                          multipoles_.data(),
                                          domain.layout().data(),
                                          domain.startCell(),
                                          domain.endCell(),
                                          d.x.data(),
                                          d.y.data(),
                                          d.z.data(),
                                          d.h.data(),
                                          d.m.data(),
                                          d.g,
                                          d.grad_P_x.data(),
                                          d.grad_P_y.data(),
                                          d.grad_P_z.data());
    }

    const MType* multipoles() const { return multipoles_.data(); }

private:
    std::vector<MType> multipoles_;
};

template<class MType, class KeyType, class Tc, class Tm, class Tf>
class MultipoleHolderGpu
{
public:
    MultipoleHolderGpu() = default;

    template<class Dataset, class Domain>
    void upsweep(const Dataset& d, const Domain& domain)
    {
        //! includes tree plus associated information, like peer ranks, assignment, counts, centers, etc
        const auto& focusTree = domain.focusTree();
        //! the focused octree, structure only
        const cstone::Octree<KeyType>& octree = focusTree.octree();

        reallocate(multipoles_, octree.numTreeNodes(), 1.05);

        mHolder_.upsweep(d.devPtrs.d_x,
                         d.devPtrs.d_y,
                         d.devPtrs.d_z,
                         d.devPtrs.d_m,
                         domain.globalTree(),
                         domain.focusTree(),
                         domain.layout().data(),
                         multipoles_.data());
    }

    template<class Dataset, class Domain>
    void traverse(Dataset& d, const Domain& domain)
    {
        d.egrav = mHolder_.compute(domain.startIndex(),
                                   domain.endIndex(),
                                   d.devPtrs.d_x,
                                   d.devPtrs.d_y,
                                   d.devPtrs.d_z,
                                   d.devPtrs.d_m,
                                   d.devPtrs.d_h,
                                   d.g,
                                   d.devPtrs.d_grad_P_x,
                                   d.devPtrs.d_grad_P_y,
                                   d.devPtrs.d_grad_P_z);
#ifdef USE_CUDA
        size_t sizeWithHalos = d.x.size();
        size_t size_np_T     = sizeWithHalos * sizeof(decltype(d.grad_P_x[0]));
        CHECK_CUDA_ERR(cudaMemcpy(d.grad_P_x.data(), d.devPtrs.d_grad_P_x, size_np_T, cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERR(cudaMemcpy(d.grad_P_y.data(), d.devPtrs.d_grad_P_y, size_np_T, cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERR(cudaMemcpy(d.grad_P_z.data(), d.devPtrs.d_grad_P_z, size_np_T, cudaMemcpyDeviceToHost));
#endif
    }

    const MType* multipoles() const { return multipoles_.data(); }

private:
    ryoanji::MultipoleHolder<Tc, Tm, Tf, KeyType, MType> mHolder_;
    std::vector<MType>                                   multipoles_;
};

} // namespace sphexa
