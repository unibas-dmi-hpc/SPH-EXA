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
        maxAccSq = cstone::maxNormSquareGpu(rawPtr(d.devData.x) + first, rawPtr(d.devData.y) + first,
                                            rawPtr(d.devData.z) + first, last - first);
    }
    else
    {
#pragma omp parallel for reduction(max : maxAccSq)
        for (size_t i = first; i < last; ++i)
        {
            cstone::Vec3<T> X{d.x[i], d.y[i], d.z[i]};
            maxAccSq = std::max(norm2(X), maxAccSq);
        }
    }

    return d.etaAcc * std::sqrt(d.eps / std::sqrt(maxAccSq));
}

template<class Dataset>
void computeTimestep(size_t first, size_t last, Dataset& d)
{
    using T = typename Dataset::RealType;

    T minDtAcc = (d.g != 0.0) ? accelerationTimestep(first, last, d) : INFINITY;

    T minDtLoc = std::min({minDtAcc, d.minDtCourant, d.maxDtIncrease * d.minDt});

    T minDtGlobal;
    MPI_Allreduce(&minDtLoc, &minDtGlobal, 1, MpiType<T>{}, MPI_MIN, MPI_COMM_WORLD);

    d.ttot += minDtGlobal;

    d.minDt_m1 = d.minDt;
    d.minDt    = minDtGlobal;
}

} // namespace sph
