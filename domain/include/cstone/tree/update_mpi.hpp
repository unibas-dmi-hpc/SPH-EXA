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
 * @brief  MPI extension for calculating distributed cornerstone octrees
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <mpi.h>

#include "cstone/primitives/mpi_wrappers.hpp"
#include "cstone/tree/csarray.hpp"
#include "cstone/tree/octree.hpp"

namespace cstone
{

/*! @brief perform one octree update, consisting of one rebalance and one node counting step
 *
 * See documentation of updateOctree
 */
template<class KeyType>
bool updateOctreeGlobal(const KeyType* keyStart,
                        const KeyType* keyEnd,
                        unsigned bucketSize,
                        std::vector<KeyType>& tree,
                        std::vector<unsigned>& counts)
{
    unsigned maxCount = std::numeric_limits<unsigned>::max();

    bool converged = updateOctree(keyStart, keyEnd, bucketSize, tree, counts, maxCount);

    std::vector<unsigned> counts_reduced(counts.size());
    MPI_Allreduce(counts.data(), counts_reduced.data(), counts.size(), MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < counts.size(); ++i)
    {
        counts[i] = std::max(counts[i], counts_reduced[i]);
    }

    return converged;
}

/*! @brief global update step of an octree, including regeneration of the internal node structure
 *
 * @tparam        KeyType     unsigned 32- or 64-bit integer
 * @param[in]     keyStart    first particle key
 * @param[in]     keyEnd      last particle key
 * @param[in]     bucketSize  max number of particles per leaf
 * @param[inout]  tree        a fully linked octree
 * @param[inout]  counts      leaf node particle counts
 * @param[in]     numRanks    number of MPI ranks
 * @return                    true if tree was not changed
 */
template<class KeyType>
bool updateOctreeGlobal(const KeyType* keyStart,
                        const KeyType* keyEnd,
                        unsigned bucketSize,
                        Octree<KeyType>& tree,
                        std::vector<unsigned>& counts)
{
    unsigned maxCount = std::numeric_limits<unsigned>::max();
    bool converged    = tree.rebalance(bucketSize, counts);
    counts.resize(tree.numLeafNodes());
    computeNodeCounts(tree.treeLeaves().data(), counts.data(), tree.numLeafNodes(), keyStart, keyEnd, maxCount, true);
    std::vector<unsigned> counts_reduced(counts.size());
    MPI_Allreduce(counts.data(), counts_reduced.data(), counts.size(), MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < counts.size(); ++i)
    {
        counts[i] = std::max(counts[i], counts_reduced[i]);
    }

    return converged;
}

} // namespace cstone
