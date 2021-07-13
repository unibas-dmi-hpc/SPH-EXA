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
 * @brief Test global octree build together with domain particle exchange
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <mpi.h>
#include <gtest/gtest.h>

#include "cstone/tree/octree_mpi.hpp"
#include "cstone/tree/octree_util.hpp"
#include "cstone/domain/layout.hpp"
#include "cstone/domain/domaindecomp_mpi.hpp"

#include "coord_samples/random.hpp"

using namespace cstone;

/*! @brief integration test between global octree build and domain particle exchange
 *
 * This test performs the following steps on each rank:
 *
 * 1. Create numParticles randomly
 *
 *    RandomGaussianCoordinates<T, KeyType> coords(numParticles, box, thisRank):
 *    Creates nParticles in the [-1,1]^3 box with random gaussian distribution
 *    centered at (0,0,0), calculate the Morton code for each particle,
 *    reorder codes and x,y,z array of coords according to ascending Morton codes
 *
 * 2. Compute global octree and node particle counts
 *
 *    auto [tree, counts] = computeOctreeGlobal(...)
 *
 * 3. split & assign a part of the global octree to each rank
 *
 * 4. Exchange particles, such that each rank gets all the particles present on all nodes
 *    that lie within the area of the octree that each rank got assigned.
 *
 * Post exchange:
 *
 * 5. Repeat 2., global octree and counts should stay the same
 *
 * 6. Repeat 3., the decomposition should now indicate that all particles stay on the
 *    node they currently are and that no rank sends any particles to other ranks.
 */
template<class KeyType, class T>
void globalRandomGaussian(int thisRank, int numRanks)
{
    int numParticles = 1000;
    int bucketSize = 64;

    Box<T> box{-1, 1};
    RandomGaussianCoordinates<T, KeyType> coords(numParticles, box, thisRank);

    std::vector<KeyType> tree = makeRootNodeTree<KeyType>();
    std::vector<unsigned> counts{numRanks * unsigned(numParticles)};

    while (!updateOctreeGlobal(coords.mortonCodes().data(), coords.mortonCodes().data() + numParticles, bucketSize, tree,
                               counts))
    {
    }

    std::vector<LocalParticleIndex> ordering(numParticles);
    // particles are in Morton order
    std::iota(begin(ordering), end(ordering), 0);

    auto assignment = singleRangeSfcSplit(counts, numRanks);
    auto sendList = createSendList<KeyType>(assignment, tree, coords.mortonCodes());

    EXPECT_EQ(std::accumulate(begin(counts), end(counts), std::size_t(0)), numParticles * numRanks);

    std::vector<T> x = coords.x();
    std::vector<T> y = coords.y();
    std::vector<T> z = coords.z();

    int numParticlesAssigned = assignment.totalCount(thisRank);

    reallocate(std::max(numParticlesAssigned, numParticles), x, y, z);
    auto [particleStart, particleEnd] =
        exchangeParticles<T>(sendList, Rank(thisRank), 0, numParticles, x.size(), numParticlesAssigned,
                             ordering.data(), x.data(), y.data(), z.data());

    /// post-exchange test:
    /// if the global tree build and assignment is repeated, no particles are exchanged anymore

    EXPECT_EQ(particleEnd - particleStart, numParticlesAssigned);

    std::vector<KeyType> newCodes(numParticlesAssigned);
    computeMortonCodes(begin(x) + particleStart, begin(x) + particleEnd,
                       begin(y) + particleStart, begin(z) + particleStart, begin(newCodes), box);

    // received particles are not stored in morton order after the exchange
    ordering.resize(numParticlesAssigned);
    std::iota(begin(ordering), end(ordering), 0);
    sort_by_key(begin(newCodes), end(newCodes), begin(ordering));

    std::vector<KeyType> newTree = makeRootNodeTree<KeyType>();
    std::vector<unsigned> newCounts{unsigned(newCodes.size())};
    while (!updateOctreeGlobal(newCodes.data(), newCodes.data() + newCodes.size(), bucketSize, newTree, newCounts))
        ;

    // global tree and counts stay the same
    EXPECT_EQ(tree, newTree);
    EXPECT_EQ(counts, newCounts);

    auto newSendList = createSendList<KeyType>(assignment, newTree, newCodes);

    for (int rank = 0; rank < numRanks; ++rank)
    {
        // the new send list now indicates that all elements on the current rank
        // stay where they are
        if (rank == thisRank) EXPECT_EQ(newSendList[rank].totalCount(), numParticlesAssigned);
        // no particles are sent to other ranks
        else
            EXPECT_EQ(newSendList[rank].totalCount(), 0);
    }
}

TEST(GlobalTreeDomain, randomGaussian)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    globalRandomGaussian<unsigned, double>(rank, nRanks);
    globalRandomGaussian<uint64_t, double>(rank, nRanks);
    globalRandomGaussian<unsigned, float>(rank, nRanks);
    globalRandomGaussian<uint64_t, float>(rank, nRanks);
}