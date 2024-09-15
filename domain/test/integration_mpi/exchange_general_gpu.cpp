/*! @file
 * @brief Test global octree build together with domain particle exchange
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <mpi.h>
#include <gtest/gtest.h>

#define USE_CUDA

#include "cstone/cuda/device_vector.h"
#include "cstone/focus/octree_focus_mpi.hpp"
#include "cstone/focus/source_center.hpp"
#include "cstone/traversal/peers.hpp"

#include "coord_samples/random.hpp"
#include "cstone/util/reallocate.hpp"

using namespace cstone;

//! @brief see test description of CPU version
template<class KeyType, class T>
static void generalExchangeRandomGaussian(int thisRank, int numRanks)
{
    const LocalIndex numParticles = 1000;
    unsigned bucketSize           = 64;
    unsigned bucketSizeLocal      = 16;
    float theta                   = 1.0;
    float invThetaEff             = invThetaMinMac(theta);

    Box<T> box{-1, 1};

    // ******************************
    // identical data on all ranks

    // common pool of coordinates, identical on all ranks
    RandomGaussianCoordinates<T, SfcKind<KeyType>> coords(numRanks * numParticles, box);

    auto [tree, counts] = computeOctree(coords.particleKeys().data(),
                                        coords.particleKeys().data() + coords.particleKeys().size(), bucketSize);

    Octree<KeyType> domainTree;
    domainTree.update(tree.data(), nNodes(tree));

    auto assignment = makeSfcAssignment(numRanks, counts, tree.data());

    // *******************************

    auto peers = findPeersMac(thisRank, assignment, domainTree, box, invThetaEff);

    KeyType focusStart = assignment[thisRank];
    KeyType focusEnd   = assignment[thisRank + 1];

    // locate particles assigned to thisRank
    auto firstAssignedIndex = findNodeAbove(coords.particleKeys().data(), coords.particleKeys().size(), focusStart);
    auto lastAssignedIndex  = findNodeAbove(coords.particleKeys().data(), coords.particleKeys().size(), focusEnd);

    // extract a slice of the common pool, each rank takes a different slice, but all slices together
    // are equal to the common pool
    std::vector<T> x(coords.x().begin() + firstAssignedIndex, coords.x().begin() + lastAssignedIndex);
    std::vector<T> y(coords.y().begin() + firstAssignedIndex, coords.y().begin() + lastAssignedIndex);
    std::vector<T> z(coords.z().begin() + firstAssignedIndex, coords.z().begin() + lastAssignedIndex);

    // Now build the focused tree using distributed algorithms. Each rank only uses its slice.
    std::vector<KeyType> particleKeys(lastAssignedIndex - firstAssignedIndex);
    computeSfcKeys(x.data(), y.data(), z.data(), sfcKindPointer(particleKeys.data()), x.size(), box);

    DeviceVector<KeyType> d_keys = particleKeys;
    DeviceVector<T> d_scratch;
    gsl::span<const KeyType> d_keysView{rawPtr(d_keys), d_keys.size()};

    DeviceVector<KeyType> d_globTree = tree;
    gsl::span<const KeyType> d_globTreeView{rawPtr(d_globTree), d_globTree.size()};
    DeviceVector<unsigned> d_globCounts = counts;
    gsl::span<const unsigned> d_globCountsView{rawPtr(d_globCounts), d_globCounts.size()};

    FocusedOctree<KeyType, T, GpuTag> focusTree(thisRank, numRanks, bucketSizeLocal);
    focusTree.converge(box, d_keysView, peers, assignment, d_globTreeView, d_globCountsView, invThetaEff, d_scratch);

    auto d_countsView = focusTree.countsAcc();
    std::vector<unsigned> testCounts(d_countsView.size());
    memcpyD2H(d_countsView.data(), d_countsView.size(), testCounts.data());

    auto octreeView = focusTree.octreeViewAcc();
    std::vector<KeyType> prefixes(octreeView.numNodes);
    memcpyD2H(octreeView.prefixes, octreeView.numNodes, prefixes.data());

    {
        for (size_t i = 0; i < testCounts.size(); ++i)
        {
            KeyType nodeStart = decodePlaceholderBit(prefixes[i]);
            KeyType nodeEnd   = nodeStart + nodeRange<KeyType>(decodePrefixLength(prefixes[i]) / 3);

            unsigned referenceCount = calculateNodeCount(nodeStart, nodeEnd, coords.particleKeys().data(),
                                                         coords.particleKeys().data() + coords.particleKeys().size(),
                                                         std::numeric_limits<unsigned>::max());
            EXPECT_EQ(testCounts[i], referenceCount);
        }
    }

    EXPECT_EQ(testCounts[0], numRanks * numParticles);
}

TEST(GeneralFocusExchangeGpu, randomGaussian)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    generalExchangeRandomGaussian<uint64_t, double>(rank, nRanks);
}
