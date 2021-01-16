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

/*! \file
 * \brief compute global minima and maxima of array ranges
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * TODO: use OpenMP parallel reduction for the local part
 */

#pragma once

#include <mpi.h>

#include "cstone/box.hpp"
#include "cstone/mpi_wrappers.hpp"


namespace sphexa
{

//! \brief compute global minimum of an array range
template<class Iterator>
auto globalMin(Iterator start, Iterator end)
{
    using T = std::decay_t<decltype(*start)>;

    T ret = *std::min_element(start, end);

    MPI_Allreduce(MPI_IN_PLACE, &ret, 1, MpiType<T>{}, MPI_MIN, MPI_COMM_WORLD);

    return ret;
}

//! \brief compute global maximum of an array range
template<class Iterator>
auto globalMax(Iterator start, Iterator end)
{
    using T = std::decay_t<decltype(*start)>;

    T ret = *std::max_element(start, end);

    MPI_Allreduce(MPI_IN_PLACE, &ret, 1, MpiType<T>{}, MPI_MAX, MPI_COMM_WORLD);

    return ret;
}

//! \brief compute global bounding box for local x,y,z arrays
template<class Iterator>
auto makeGlobalBox(Iterator xB,
                   Iterator xE,
                   Iterator yB,
                   Iterator zB,
                   bool pbcX = false,
                   bool pbcY = false,
                   bool pbcZ = false)
{
    using T = std::decay_t<decltype(*xB)>;

    int nElements = xE - xB;
    return Box<T>{globalMin(xB, xE), globalMax(xB, xE),
                  globalMin(yB, yB + nElements), globalMax(yB, yB + nElements),
                  globalMin(zB, zB + nElements), globalMax(zB, zB + nElements),
                  pbcX, pbcY, pbcZ};
};

} // namespace sphexa