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
 * \brief  MPI extension for calculating distributed cornerstone octrees
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 */


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
