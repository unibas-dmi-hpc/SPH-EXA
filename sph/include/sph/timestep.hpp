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
 * @brief Min-reduction to determine global timestep
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 * @author Aurelien Cavelan
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <vector>
#include <mpi.h>

#include "cstone/primitives/primitives_gpu.h"
#include "kernels.hpp"

namespace sph
{

//! @brief limit time-step based on accelerations when gravity is enabled
template<class Dataset>
auto accelerationTimestep(size_t first, size_t last, const Dataset& d)
{
    using T = typename Dataset::RealType;

    T maxAccSq = 0.0;
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        maxAccSq = cstone::maxNormSquareGpu(rawPtr(d.devData.ax) + first, rawPtr(d.devData.ay) + first,
                                            rawPtr(d.devData.az) + first, last - first);
    }
    else
    {
#pragma omp parallel for reduction(max : maxAccSq)
        for (size_t i = first; i < last; ++i)
        {
            cstone::Vec3<T> X{d.ax[i], d.ay[i], d.az[i]};
            maxAccSq = std::max(norm2(X), maxAccSq);
        }
    }

    return d.etaAcc * std::sqrt(d.eps / std::sqrt(maxAccSq));
}

//! @brief limit time-step based on divergence of velocity, this is called in the propagator when Divv is available
template<class Dataset>
auto rhoTimestep(size_t first, size_t last, const Dataset& d)
{
    using T = std::decay_t<decltype(d.divv[0])>;

    T maxDivv = -INFINITY;
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        if (d.devData.divv.empty()) { throw std::runtime_error("Divv needs to be available in rhoTimestep\n"); }
        auto minmax = cstone::MinMaxGpu<T>{}(rawPtr(d.devData.divv) + first, rawPtr(d.devData.divv) + last);
        maxDivv     = std::get<1>(minmax);
    }
    else
    {
        if (d.divv.empty()) { throw std::runtime_error("Divv needs to be available in rhoTimestep\n"); }

#pragma omp parallel for reduction(max : maxDivv)
        for (size_t i = first; i < last; ++i)
        {
            maxDivv = std::max(d.divv[i], maxDivv);
        }
    }
    return d.Krho / std::abs(maxDivv);
}

template<class Dataset, class... Ts>
void computeTimestep(size_t first, size_t last, Dataset& d, Ts... extraTimesteps)
{
    using T = typename Dataset::RealType;

    T minDtAcc = (d.g != 0.0) ? accelerationTimestep(first, last, d) : INFINITY;

    T minDtLoc = std::min({minDtAcc, d.minDtCourant, d.minDtRho, d.maxDtIncrease * d.minDt, extraTimesteps...});

    T minDtGlobal;
    MPI_Allreduce(&minDtLoc, &minDtGlobal, 1, MpiType<T>{}, MPI_MIN, MPI_COMM_WORLD);

    d.ttot += minDtGlobal;

    d.minDt_m1 = d.minDt;
    d.minDt    = minDtGlobal;
}

//! @brief compute Divv-limited timestep for each group when block time-steps are active
template<class Dataset>
void groupDivvTimestep(const GroupView& grp, float* groupDt, const Dataset& d)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        groupDivvTimestepGpu(d.Krho, grp, rawPtr(d.devData.divv), groupDt);
    }
}

//! @brief compute acceleration-limited timestep for each group when block time-steps are active
template<class Dataset>
void groupAccTimestep(const GroupView& grp, float* groupDt, const Dataset& d)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        groupAccTimestepGpu(d.etaAcc * std::sqrt(d.eps), grp, rawPtr(d.devData.ax), rawPtr(d.devData.ay),
                            rawPtr(d.devData.az), groupDt);
    }
}

//! @brief sort groupDt, keeping track of the ordering
template<class AccVec>
void sortGroupDt(float* groupDt, cstone::LocalIndex* groupIndices, cstone::LocalIndex numGroups, AccVec& scratch)
{
    using cstone::LocalIndex;
    size_t oldSize  = reallocateBytes(scratch, (sizeof(float) + sizeof(LocalIndex)) * numGroups);
    auto*  keyBuf   = reinterpret_cast<float*>(rawPtr(scratch));
    auto*  valueBuf = reinterpret_cast<LocalIndex*>(keyBuf + numGroups);
    cstone::sequenceGpu(groupIndices, numGroups, 0u);
    cstone::sortByKeyGpu(groupDt, groupDt + numGroups, groupIndices, keyBuf, valueBuf);
    reallocate(oldSize, scratch);
};

//! @brief return the local minimum timestep and the biggest timestep of the fastest fraction of paritcles
inline auto timestepRangeGpu(const float* groupDt, cstone::LocalIndex numGroups, float fastFraction)
{
    std::array<float, 2> minGroupDt;
    memcpyD2H(groupDt, 1, minGroupDt.data());
    memcpyD2H(groupDt + cstone::LocalIndex(fastFraction * numGroups), 1, minGroupDt.data() + 1);
    return minGroupDt;
}

//! @brief extract the specified subgroup [first:last] indexed through @p index from @p grp into @p outGroup
template<class Accelerator>
inline void extractGroupGpu(const GroupView& grp, const cstone::LocalIndex* indices, cstone::LocalIndex first,
                            cstone::LocalIndex last, GroupData<Accelerator>& out)
{
    auto numOutGroups = last - first;
    reallocate(out.data, 2 * numOutGroups, 1.01);

    out.firstBody  = 0;
    out.lastBody   = 0;
    out.numGroups  = numOutGroups;
    out.groupStart = rawPtr(out.data);
    out.groupEnd   = rawPtr(out.data) + numOutGroups;

    cstone::gatherGpu(indices + first, numOutGroups, grp.groupStart, out.groupStart);
    cstone::gatherGpu(indices + first, numOutGroups, grp.groupEnd, out.groupEnd);
}

struct Timestep
{
    static constexpr int maxNumRungs = 4;
    //! @brief maxDt = minDt * 2^numRungs;
    float minDt;
    int   numRungs{1};
    //! @brief 0,...,2^numRungs
    int substep{0};

    std::array<cstone::LocalIndex, maxNumRungs + 1> rungRanges;

    template<class Archive>
    void loadOrStore(Archive* ar, const std::string& prefix)
    {
        ar->stepAttribute(prefix + "minDt", &minDt, 1);
        ar->stepAttribute(prefix + "numRungs", &numRungs, 1);
        ar->stepAttribute(prefix + "substep", &substep, 1);
    }
};

//! @brief Determine timestep rungs
template<class AccVec>
Timestep computeGroupTimestep(const GroupView& grp, float* groupDt, cstone::LocalIndex* groupIndices, AccVec& scratch)
{
    using cstone::LocalIndex;

    std::array<float, 2> minGroupDt;
    if constexpr (IsDeviceVector<AccVec>{})
    {
        sortGroupDt(groupDt, groupIndices, grp.numGroups, scratch);
        minGroupDt = timestepRangeGpu(groupDt, grp.numGroups, 0.4);
    }

    std::array<float, 2> minDtGlobal;
    mpiAllreduce(minGroupDt.data(), minDtGlobal.data(), minGroupDt.size(), MPI_MIN);

    int numRungs = std::min(int(log2(minDtGlobal[1] / minDtGlobal[0])) + 1, Timestep::maxNumRungs);

    // find ranges of 2*minDt, 4*minDt, 8*minDt
    // groupDt is sorted, groups belonging to a specific rung will correspond to index ranges
    std::array<LocalIndex, Timestep::maxNumRungs + 1> rungRanges{0};
    std::fill(rungRanges.begin() + 1, rungRanges.end(), grp.numGroups);
    if constexpr (IsDeviceVector<AccVec>{})
    {
        for (int rung = 1; rung < numRungs; ++rung)
        {
            float maxDtRung  = (1 << rung) * minDtGlobal[0];
            rungRanges[rung] = cstone::lowerBoundGpu(groupDt, groupDt + grp.numGroups, maxDtRung);
        }
    }

    return {.minDt = minDtGlobal[0], .numRungs = numRungs, .substep = 0, .rungRanges = rungRanges};
}

} // namespace sph
