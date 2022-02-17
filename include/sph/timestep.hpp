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

#include <vector>
#include <math.h>
#include <algorithm>

#include "kernels.hpp"

#ifdef USE_MPI
#include "mpi.h"
#endif

namespace sphexa
{
namespace sph
{

template<class T1, class T2, class T3>
auto tsKCourant(T1 maxvsignal, T2 h, T3 c, double kcour)
{
    using T = std::common_type_t<T1, T2, T3>;
    T v = maxvsignal > T(0) ? maxvsignal : c;
    return T(kcour * h / v);
}

template<class Dataset>
auto computeMinTimestepImpl(size_t startIndex, size_t endIndex, Dataset& d)
{
    using T = typename Dataset::RealType;
    T minDt = INFINITY;

#pragma omp parallel for reduction(min : minDt)
    for (size_t i = startIndex; i < endIndex; i++)
    {
        T dt_i = tsKCourant(d.maxvsignal[i], d.h[i], d.c[i], d.Kcour);
        minDt  = std::min(minDt, dt_i);
    }

    return minDt;
}

template<class Dataset>
void computeTimestep(size_t startIndex, size_t endIndex, Dataset& d)
{
    using T = typename Dataset::RealType;
    T minDt = computeMinTimestepImpl(startIndex, endIndex, d);
    minDt   = std::min(minDt, d.maxDtIncrease * d.minDt);

#ifdef USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, &minDt, 1, MpiType<T>{}, MPI_MIN, MPI_COMM_WORLD);
#endif

#pragma omp parallel for schedule(static)
    for (size_t i = startIndex; i < endIndex; i++)
    {
        d.dt[i] = minDt;
    }

    d.ttot += minDt;
    d.minDt = minDt;
}

} // namespace sph
} // namespace sphexa
