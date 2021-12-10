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
 * @brief compute global minima and maxima of array ranges
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 */

#pragma once

#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <cmath>

#include "box.hpp"
#include "cstone/primitives/mpi_wrappers.hpp"


namespace cstone
{

//! @brief compute global minimum of an array range
template<class Iterator>
auto globalMin(Iterator start, Iterator end)
{
    assert(end > start);
    using T = std::decay_t<decltype(*start)>;

    T minimum = INFINITY;

    #pragma omp parallel for reduction(min : minimum)
    for (size_t pi = 0; pi < std::size_t(end-start); pi++)
    {
        T value = start[pi];
        minimum = std::min(minimum, value);
    }

    MPI_Allreduce(MPI_IN_PLACE, &minimum, 1, MpiType<T>{}, MPI_MIN, MPI_COMM_WORLD);

    return minimum;
}

//! @brief compute global maximum of an array range
template<class Iterator>
auto globalMax(Iterator start, Iterator end)
{
    assert(end > start);
    using T = std::decay_t<decltype(*start)>;

    T maximum = -INFINITY;

    #pragma omp parallel for reduction(max : maximum)
    for (size_t pi = 0; pi < std::size_t(end-start); pi++)
    {
        T value = start[pi];
        maximum = std::max(maximum, value);
    }

    MPI_Allreduce(MPI_IN_PLACE, &maximum, 1, MpiType<T>{}, MPI_MAX, MPI_COMM_WORLD);

    return maximum;
}

/*! @brief compute global bounding box for local x,y,z arrays
 *
 * @tparam Iterator      coordinate array iterator, providing random access
 * @param  xB            x coordinate array start
 * @param  xE            x coordinate array end
 * @param  yB            y coordinate array start
 * @param  zB            z coordinate array start
 * @param  previousBox   previous coordinate bounding box, default non-pbc box
 *                       with limits ignored
 * @return               the new bounding box
 *
 * For each periodic dimension, limits are fixed and will not be modified.
 * For non-periodic dimensions, limits are determined by global min/max.
 */
template<class Iterator>
auto makeGlobalBox(Iterator xB,
                   Iterator xE,
                   Iterator yB,
                   Iterator zB,
                   const Box<std::decay_t<decltype(*xB)>>& previousBox =
                       Box<std::decay_t<decltype(*xB)>>{0,1})
{
    using T = std::decay_t<decltype(*xB)>;

    std::size_t nElements = xE - xB;
    T newXmin = (previousBox.pbcX()) ? previousBox.xmin() : globalMin(xB, xB + nElements);
    T newYmin = (previousBox.pbcY()) ? previousBox.ymin() : globalMin(yB, yB + nElements);
    T newZmin = (previousBox.pbcZ()) ? previousBox.zmin() : globalMin(zB, zB + nElements);
    T newXmax = (previousBox.pbcX()) ? previousBox.xmax() : globalMax(xB, xB + nElements);
    T newYmax = (previousBox.pbcY()) ? previousBox.ymax() : globalMax(yB, yB + nElements);
    T newZmax = (previousBox.pbcZ()) ? previousBox.zmax() : globalMax(zB, zB + nElements);

    return Box<T>{newXmin, newXmax, newYmin, newYmax, newZmin, newZmax,
                  previousBox.pbcX(), previousBox.pbcY(), previousBox.pbcZ()};
};

} // namespace cstone

