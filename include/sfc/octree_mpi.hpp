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
 * \tparam I           32- or 64-bit unsigned integer type
 * \param codesStart   particle morton code sequence start
 * \param codesEnd     particle morton code sequence end
 * \param bucketSize   maximum number of particles/codes per octree leaf node
 * \param sfcRanges    (i,i+1) pairs of Morton code ranges assigned to rank
 * \param nRanges      number of Morton code pairs in \a nodeRanges
 * \param[inout] tree  initial tree for the first iteration
 * \return             the tree and the node counts
 */
template <class I>
std::tuple<std::vector<I>, std::vector<std::size_t>> computeOctreeGlobal(const I* codesStart, const I* codesEnd, int bucketSize,
                                                                         const I* sfcRanges, int nRanges,
                                                                         std::vector<I>&& tree = std::vector<I>(0))
{
    return computeOctree<I, GlobalReduce>(codesStart, codesEnd, bucketSize, sfcRanges, nRanges, std::move(tree));
}

/*! \brief Compute the global maximum value of a given input array for each node in the global or local octree
 *
 * Example: For each node, compute the maximum smoothing length of all particles in that node
 *
 * \tparam I           32- or 64-bit unsigned integer type
 * \param tree         octree nodes given as Morton codes of length @a nNodes+1
 *                     This function does not rely on octree invariants, sortedness of the nodes
 *                     is the only requirement.
 * \param nNodes       number of nodes in tree
 * \param codesStart   sorted Morton code range start of particles to count
 * \param codesEnd     sorted Morton code range end of particles to count
 * \param ordering     Access input according to \a ordering
 *                     The sequence input[ordering[i]], i=0,...,N must list the elements of input
 *                     (i.e. the smoothing lengths) such that input[i] is a property of the particle
 *                     (x[i], y[i], z[i]), with x,y,z sorted according to Morton ordering.
 * \param input        Array to compute maximum over nodes, length = codesEnd - codesStart
 * \param output       maximum per node, length = @a nNodes
 */
template <class I, class T>
void computeNodeMaxGlobal(const I *tree, int nNodes, const I *codesStart, const I *codesEnd, const int *ordering, const T *input, T *output)
{
    computeNodeMax(tree, nNodes, codesStart, codesEnd, ordering, input, output);
    MPI_Allreduce(MPI_IN_PLACE, output, nNodes, MpiType<T>{}, MPI_MAX, MPI_COMM_WORLD);
}

} // namespace sphexa
