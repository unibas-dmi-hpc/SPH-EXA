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
#include "cstone/domain/domaindecomp_mpi.hpp"
#include "cstone/domain/peers.hpp"
#include "cstone/domain/layout.hpp"
#include "cstone/tree/octree_focus_mpi.hpp"
#include "cstone/tree/octree_util.hpp"

#include "coord_samples/random.hpp"

using namespace cstone;

/*! @brief
 *
 */
template<class KeyType, class T>
void focusDomain(int thisRank, int numRanks)
{
    size_t numParticles = 1000;
    unsigned bucketSize = 64;
    unsigned bucketSizeLocal = 16;
    float theta = 1.0;

    Box<T> box{-1, 1};

    std::vector<T> x, y, z;
    std::vector<KeyType> particleKeys(numParticles);
    RandomGaussianCoordinates<T, KeyType> coords(numParticles, box, thisRank);
    x = coords.x();
    y = coords.y();
    z = coords.z();

    computeMortonCodes(begin(x), end(x), begin(y), begin(z), begin(particleKeys), box);

    // Now build the trees, using distributed algorithms. Each rank only has its slice, the common pool is gone.

    std::vector<KeyType> tree = makeRootNodeTree<KeyType>();
    std::vector<unsigned> counts{unsigned(numParticles) * numRanks};
    while (!updateOctreeGlobal(particleKeys.data(), particleKeys.data() + numParticles, bucketSize, tree, counts))
        ;

    std::vector<LocalParticleIndex> ordering(numParticles);
    // particles are in Morton order
    std::iota(begin(ordering), end(ordering), 0);

    auto assignment = singleRangeSfcSplit(counts, numRanks);
    auto sendList = createSendList<KeyType>(assignment, tree, particleKeys);

    LocalParticleIndex nParticlesAssigned = assignment.totalCount(thisRank);

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
}

TEST(GlobalTreeDomain, randomGaussian)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    focusDomain<unsigned, double>(rank, nRanks);
    // focusDomain<uint64_t, double>(rank, nRanks);
    // focusDomain<unsigned, float>(rank, nRanks);
    // focusDomain<uint64_t, float>(rank, nRanks);
}