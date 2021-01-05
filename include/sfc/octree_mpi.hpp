#pragma once

#include <mpi.h>

#include "sfc/mpi_wrappers.hpp"
#include "sfc/octree.hpp"

namespace sphexa
{

struct GlobalReduce
{
    void operator()(std::vector<std::size_t> &counts)
    {
        MPI_Allreduce(MPI_IN_PLACE, counts.data(), counts.size(), MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
    }
};

/*! \brief compute an octree from morton codes for a specified bucket size
 *
 * See documentation of computeOctree
 */
template <class I>
std::tuple<std::vector<I>, std::vector<std::size_t>> computeOctreeGlobal(const I *codesStart, const I *codesEnd, int bucketSize,
                                                                         std::vector<I> &&tree = std::vector<I>(0))
{
    return computeOctree<I, GlobalReduce>(codesStart, codesEnd, bucketSize, std::move(tree));
}

/*! \brief Compute the global maximum value of a given input array for each node in the global or local octree
 *
 * See documentation of computeNodeMax
 */
template <class I, class T>
void computeNodeMaxGlobal(const I *tree, int nNodes, const I *codesStart, const I *codesEnd, const int *ordering, const T *input, T *output)
{
    computeNodeMax(tree, nNodes, codesStart, codesEnd, ordering, input, output);
    MPI_Allreduce(MPI_IN_PLACE, output, nNodes, MpiType<T>{}, MPI_MAX, MPI_COMM_WORLD);
}

} // namespace sphexa
