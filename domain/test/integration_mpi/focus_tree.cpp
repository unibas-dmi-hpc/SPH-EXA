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
#include "cstone/domain/layout.hpp"
#include "cstone/domain/domaindecomp_mpi.hpp"
#include "cstone/domain/peers.hpp"
#include "cstone/tree/octree_focus_mpi.hpp"
#include "cstone/tree/octree_util.hpp"

#include "coord_samples/random.hpp"

using namespace cstone;

template<class KeyType, class T>
FocusedOctree<KeyType> createReferenceFocusTree(const Box<T>& box, gsl::span<const KeyType> particleKeys, int myRank,
                                                int numRanks, unsigned bucketSize, unsigned bucketSizeLocal,
                                                float theta)
{
    auto [tree, counts] = computeOctree(particleKeys.data(), particleKeys.data() + particleKeys.size(), bucketSize);

    Octree<KeyType> domainTree;
    domainTree.update(begin(tree), end(tree));

    auto assignment = singleRangeSfcSplit(counts, numRanks);

    FocusedOctree<KeyType> focusTree(bucketSizeLocal, theta);
    while (!focusTree.update(box, particleKeys, tree[assignment.firstNodeIdx(myRank)],
                             tree[assignment.lastNodeIdx(myRank)], {}))
    {
    }

    return focusTree;
}

/*! @brief test for particle-count-exchange of distributed focused octrees
 *
 * First, all ranks create numRanks * numParticles random gaussian particle coordinates.
 * Since the random number generator is seeded with the same value on all ranks, all of them
 * will generate exactly the same numRanks * numParticles coordinates.
 *
 * This common coordinate set is then used to build a focus tree on each rank, using
 * non-communicating local algorithms to serve as the reference.
 * Each rank then takes the <thisRank>-th fraction of the common coordinate set and discards the other coordinates,
 * such that all ranks combined still have the original common set.
 * From the distributed coordinate set, the same focused trees are then built, but with distributed communicating
 * algorithms. This should yield the same tree on each rank as the local case,
 */
template<class KeyType, class T>
void globalRandomGaussian(int thisRank, int numRanks)
{
    size_t numParticles = 1000;
    unsigned bucketSize = 64;
    unsigned bucketSizeLocal = 16;
    float theta = 1.0;

    Box<T> box{-1, 1};

    std::vector<T> x, y, z;
    std::vector<KeyType> particleKeys(numParticles);

    FocusedOctree<KeyType> referenceFocusTree(bucketSizeLocal, theta);
    {
        // common pool of coordinates, identical on all ranks
        RandomGaussianCoordinates<T, KeyType> coords(numRanks * numParticles, box);

        // reference tree built locally from all particles in the common pool, focused on the executing rank
        referenceFocusTree = createReferenceFocusTree<KeyType>(box, coords.mortonCodes(), thisRank, numRanks,
                                                               bucketSize, bucketSizeLocal, theta);

        // extract a slice of the common pool, each rank takes a different slice, but all slices together
        // are equal to the common pool
        x = std::vector<T>(coords.x().begin() + thisRank * numParticles,
                           coords.x().begin() + (thisRank + 1) * numParticles);
        y = std::vector<T>(coords.y().begin() + thisRank * numParticles,
                           coords.y().begin() + (thisRank + 1) * numParticles);
        z = std::vector<T>(coords.z().begin() + thisRank * numParticles,
                           coords.z().begin() + (thisRank + 1) * numParticles);
    }
    computeMortonCodes(begin(x), end(x), begin(y), begin(z), begin(particleKeys), box);

    // Now build the trees, using distributed algorithms. Each rank only has its slice, the common pool is gone.

    std::vector<KeyType> tree = makeRootNodeTree<KeyType>();
    std::vector<unsigned> counts{unsigned(numParticles) * numRanks};
    while (!updateOctreeGlobal(particleKeys.data(), particleKeys.data() + numParticles, bucketSize, tree, counts))
        ;

    EXPECT_EQ(numRanks * numParticles, std::accumulate(counts.begin(), counts.end(), 0lu));

    std::vector<int> ordering(numParticles);
    // particles are in Morton order
    std::iota(begin(ordering), end(ordering), 0);

    auto assignment = singleRangeSfcSplit(counts, numRanks);
    auto sendList = createSendList<KeyType>(assignment, tree, particleKeys);

    int nParticlesAssigned = assignment.totalCount(thisRank);

    reallocate(nParticlesAssigned, x, y, z);
    exchangeParticles<T>(sendList, Rank(thisRank), nParticlesAssigned, ordering.data(), x.data(), y.data(), z.data());

    reallocate(nParticlesAssigned, particleKeys);
    computeMortonCodes(begin(x), end(x), begin(y), begin(z), begin(particleKeys), box);

    // reorder arrays according to ascending SFC after exchangeParticles
    reallocate(nParticlesAssigned, ordering);
    std::iota(begin(ordering), end(ordering), 0);
    sort_by_key(begin(particleKeys), end(particleKeys), begin(ordering));
    reorder(ordering, x);
    reorder(ordering, y);
    reorder(ordering, z);

    Octree<KeyType> domainTree;
    domainTree.update(begin(tree), end(tree));
    auto peers = findPeersMac(thisRank, assignment, domainTree, box, theta);

    FocusedOctree<KeyType> focusTree(bucketSizeLocal, theta);

    int converged = 0;
    while (converged != numRanks)
    {
        converged = focusTree.updateGlobal(box, particleKeys, thisRank, peers, assignment, tree, counts);
        MPI_Allreduce(MPI_IN_PLACE, &converged, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    }

    // the locally built reference tree should be identical to the tree build with distributed particles
    EXPECT_EQ(focusTree.treeLeaves(), referenceFocusTree.treeLeaves());
    EXPECT_EQ(focusTree.leafCounts(), referenceFocusTree.leafCounts());
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