#pragma once

#include <mpi.h>

#include "sfc/box.hpp"
#include "sfc/mpi_wrappers.hpp"

/*! \brief \file compute global minima and maxima of array ranges
 *
 * TODO: use OpenMP parallel reduction for the local part
 */

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
                   Iterator yE,
                   Iterator zB,
                   Iterator zE,
                   bool pbcX = false,
                   bool pbcY = false,
                   bool pbcZ = false)
{
    using T = std::decay_t<decltype(*xB)>;

    return Box<T>{globalMin(xB, xE), globalMax(xB, xE),
                  globalMin(yB, yE), globalMax(yB, yE),
                  globalMin(zB, zE), globalMax(zB, zE),
                  pbcX, pbcY, pbcZ};
};

} // namespace sphexa