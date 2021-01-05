

#include <mpi.h>
#include <gtest/gtest.h>

#include "sfc/octree_mpi.hpp"
#include "sfc/domaindecomp_mpi.hpp"

#include "coord_samples/random.hpp"

/*! \brief integration test between global octree build and domain particle exchange
 *
 * This test performs the following steps on each rank:
 *
 * 1. Create nParticles randomly
 *
 *    RandomGaussianCoordinates<T, I> coords(nParticles, box):
 *    Creates nParticles in the [-1,1]^3 box with random gaussian distribution
 *    centered at (0,0,0), calculate the Morton code for each particle,
 *    reorder codes and x,y,z array of coords according to ascending Morton codes
 *
 * 2. Compute global octree and node particle counts
 *
 *    auto [tree, counts] = sphexa::computeOctreeGlobal(...)
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
template<class I, class T>
void globalRandomGaussian(int thisRank, int nRanks)
{
    int nParticles = 1000;
    int bucketSize = 64;

    sphexa::Box<T> box{-1, 1};
    RandomGaussianCoordinates<T, I> coords(nParticles, box);

    auto [tree, counts] =
        sphexa::computeOctreeGlobal(coords.mortonCodes().data(), coords.mortonCodes().data() + nParticles,
                                    bucketSize);

    std::vector<int> ordering(nParticles);
    // particles are in Morton order
    std::iota(begin(ordering), end(ordering), 0);

    auto assignment        = sphexa::singleRangeSfcSplit(tree, counts, nRanks);
    auto sendList          = sphexa::createSendList(assignment, coords.mortonCodes().data(),
                                                    coords.mortonCodes().data() + nParticles);

    EXPECT_EQ(std::accumulate(begin(counts), end(counts), std::size_t(0)), nParticles * nRanks);

    std::vector<T> x = coords.x();
    std::vector<T> y = coords.y();
    std::vector<T> z = coords.z();

    int nParticlesAssigned = assignment.totalCount(thisRank);

    sphexa::exchangeParticles<T>(sendList, nParticlesAssigned, thisRank, ordering.data(), x, y, z);

    /// post-exchange test:
    /// if the global tree build and assignment is repeated, no particles are exchanged anymore

    EXPECT_EQ(x.size(), nParticlesAssigned);

    std::vector<I> newCodes(nParticlesAssigned);
    sphexa::computeMortonCodes(begin(x), end(x), begin(y), begin(z), begin(newCodes), box);

    // received particles are not stored in morton order after the exchange
    ordering.resize(nParticlesAssigned);
    sphexa::sort_invert(cbegin(newCodes), cend(newCodes), begin(ordering));
    sphexa::reorder(ordering, newCodes);

    auto [newTree, newCounts] =
        sphexa::computeOctreeGlobal(newCodes.data(), newCodes.data() + newCodes.size(), bucketSize);

    // global tree and counts stay the same
    EXPECT_EQ(tree, newTree);
    EXPECT_EQ(counts, newCounts);

    auto newSendList = sphexa::createSendList(assignment, newCodes.data(), newCodes.data() + nParticlesAssigned);

    for (int rank = 0; rank < nRanks; ++rank)
    {
        // the new send list now indicates that all elements on the current rank
        // stay where they are
        if (rank == thisRank)
            EXPECT_EQ(newSendList[rank].totalCount(), nParticlesAssigned);
        // no particles are sent to other ranks
        else
            EXPECT_EQ(newSendList[rank].totalCount(), 0);
    }

    // quick check that send buffers are created w.r.t ordering
    auto xBuffer = sphexa::createSendBuffer(newSendList[thisRank], x.data(), ordering.data());
    sphexa::reorder(ordering, x);
    EXPECT_EQ(xBuffer, x);
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