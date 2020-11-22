

#include <mpi/mpi.h>
#include <gtest/gtest.h>

#define USE_MPI

#include "sfc/domaindecomp.hpp"
#include "sfc/octree_mpi.hpp"

#include "coord_samples/random.hpp"

//! \brief integration test between global octree build and domain particle exchange
template<class I, class T>
void globalRandomGaussian(int thisRank, int nRanks)
{
    int nParticles = 1000;
    int bucketSize = 64;

    sphexa::Box<T> box{-1, 1};
    RandomGaussianCoordinates<T, I> coords(nParticles, box);

    auto [tree, counts] =
        sphexa::computeOctreeGlobal(coords.mortonCodes().data(),
                                    coords.mortonCodes().data() + nParticles, bucketSize);

    std::vector<int> ordering(nParticles);
    // particles are in Morton order
    std::iota(begin(ordering), end(ordering), 0);

    auto assignment = sphexa::singleRangeSfcSplit(tree, counts, nRanks);
    auto sendList   = sphexa::createSendList(assignment, coords.mortonCodes());

    EXPECT_EQ(std::accumulate(begin(counts), end(counts), 0), nParticles * nRanks);

    std::vector<T> x = coords.x();
    std::vector<T> y = coords.y();
    std::vector<T> z = coords.z();

    std::size_t nParticlesAssigned = assignment[thisRank].count();

    sphexa::exchangeParticles<T>(sendList, nParticlesAssigned, thisRank, ordering, x, y, z);

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

    auto newSendList = sphexa::createSendList(assignment, newCodes);

    for (int rank = 0; rank < nRanks; ++rank)
    {
        // the new send list now indicates that all elements on the current rank
        // stay where they are
        if (rank == thisRank)
            EXPECT_EQ(newSendList[rank].count(), nParticlesAssigned);
        // no particles are sent to other ranks
        else
            EXPECT_EQ(newSendList[rank].count(), 0);
    }

    // quick check that send buffers are created w.r.t ordering
    auto xBuffer = sphexa::createSendBuffer(newSendList[thisRank], x, ordering);
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