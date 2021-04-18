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
 * @brief Tests that halo discovery finds all halos for a complete neighbor search
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "cstone/findneighbors.hpp"
#include "cstone/domain/halodiscovery.hpp"
#include "cstone/domain/layout.hpp"

#include "coord_samples/random.hpp"

/*! @brief @file Combines halo search with neighbor search
 *
 * We test that halo search finds all particles required for a complete
 * neighbor search.
 *
 */

using namespace cstone;

template<class T>
static void findNeighborsNaive(int i, const T* x, const T* y, const T* z, const T* h, int n,
                               int *neighbors, int *neighborsCount, int ngmax, const Box<T>& box)
{
    T r2 = h[i] * h[i];

    T xi = x[i], yi = y[i], zi = z[i];

    int ngcount = 0;
    for (int j = 0; j < n; ++j)
    {
        if (j == i) { continue; }
        // i only interacts with j if j also interacts with i
        T r2mutual = std::min(h[j] * h[j], r2);
        if (ngcount < ngmax && distanceSqPbc(xi, yi, zi, x[j], y[j], z[j], box) < r2mutual)
        {
            neighbors[i * ngmax + ngcount++] = j;
        }
    }
    neighborsCount[i] = ngcount;
}

template<class I, class T>
void extractParticles(const T* source, const I* sourceCodes, int nSourceCodes, const int* ordering,
                      const int* nodeList, const int* nodeOffsets, int nNodesPresent,
                      const I* tree, T* destination)
{
    for (int i = 0; i < nNodesPresent; ++i)
    {
        int nodeIndex = nodeList[i];

        I nodeCode = tree[nodeIndex];
        int offset    = nodeOffsets[i];
        int nodeCount = nodeOffsets[i+1] - offset;

        int sourceLocation = std::lower_bound(sourceCodes, sourceCodes + nSourceCodes, nodeCode)
                             - sourceCodes;

        for (int j = 0; j < nodeCount; ++j)
            destination[offset + j] = source[ordering[sourceLocation+j]];
    }
}

template<class I, class T>
void testExtractParticles()
{
    int nParticles = 20;
    std::vector<T> x(nParticles);
    std::iota(begin(x), end(x), 100);

    std::vector<int> ordering(nParticles);
    std::iota(begin(ordering), end(ordering), 0);

    std::vector<I> codes(nParticles);
    std::iota(begin(codes), end(codes), 0);

    //                  0 1 2 3 4 5  6  7  8  9  10
    std::vector<I> tree{0,2,4,6,8,10,12,14,16,18,20};

    std::vector<int> presentNodes{0,1,2,3,4, 7, 9};
    std::vector<int> nodeOffsets {0,2,4,6,8,10,12,14};

    std::vector<T> extrX(14);
    extractParticles(x.data(), codes.data(), x.size(), ordering.data(),
                     presentNodes.data(), nodeOffsets.data(), presentNodes.size(), tree.data(), extrX.data());

    std::vector<T> refX{100,101,102,103,104,105,106,107,108,109, 114, 115, 118, 119};
    EXPECT_EQ(refX, extrX);
}


TEST(HaloNeighbors, extractParticlesTest)
{
    testExtractParticles<unsigned, double>();
    testExtractParticles<uint64_t, double>();
    testExtractParticles<unsigned, float>();
    testExtractParticles<uint64_t, float>();
}


/*! @brief Test that halo discovery finds all nodes needed for a correct neighbor search
 *
 * @tparam I  unsigned 32- or 64-bit integer
 * @tparam T  float or double
 *
 * This test creates nParticles gaussian distributed particles in the box [-1,1]^3.
 * The resulting tree is then split into two pieces A and B.
 * From the perspective of A, halo discovery is performed to determine
 * all the nodes from B needed for a correct neighbor list for the particles in A.
 *
 * A + halos of A in B are then copied into separate x,y,z,h arrays and a neighbor search
 * is performed for the particles in A. The neighbors should exactly match the
 * neighbors of the particles in A determined with the original full A+B arrays.
 *
 * Additionally, this test verifies the mutuality property of the halo discovery,
 * where two nodes i and j are only halos if (i+maxH_i) overlaps with j AND (j+maxH_j)
 * overlaps with i. This needs to be reflected in the neighbor search algorithm, such
 * that two particles p and q only interact with each other if |r_p - r_q| < min(h_p, h_q).
 *
 * The smoothing lengths parameters in h for each particles are tuned such that there is
 * enough asymmetry for some nodes (i+maxH_i) to overlap with j, while (j+maxH_j) does
 * not overlap with i.
 */
template<class I, class T>
void randomGaussianHaloNeighbors(bool usePbc)
{
    int nParticles = 1000;
    int bucketSize = 10;
    T   smoothingL = 0.02;

    int nRanks     = 2;

    Box<T> box{-1, 1, -1, 1, -1, 1, usePbc, usePbc, usePbc};
    RandomGaussianCoordinates<T, I> coords(nParticles, box);

    std::vector<I> codes = coords.mortonCodes();

    std::vector<T> x = coords.x();
    std::vector<T> y = coords.y();
    std::vector<T> z = coords.z();
    std::vector<T> h(nParticles, smoothingL);

    // introduce asymmetry in h with nodes far from the center having a much bigger
    // interaction radius
    if (!usePbc)
    {
        for (std::size_t i = 0; i < h.size(); ++i)
        {
            h[i] = smoothingL * (0.2 + 30*(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]));
        }
    }
    else {
        h = std::vector<T>(nParticles, 0.1);
    }

    std::vector<int> ordering(nParticles);
    std::iota(begin(ordering), end(ordering), 0);

    auto [tree, counts] =
        computeOctree(coords.mortonCodes().data(), coords.mortonCodes().data() + nParticles,
                                    bucketSize);

    std::vector<T> hNode(nNodes(tree));
    computeHaloRadii(tree.data(), nNodes(tree), codes.data(), codes.data() + nParticles, ordering.data(),
                     h.data(), hNode.data());

    SpaceCurveAssignment<I> assignment = singleRangeSfcSplit(tree, counts, nRanks);

    // find halos for rank 0
    int myRank = 0;
    std::vector<pair<TreeNodeIndex>> haloPairs;

    //Box<T> box2{-1, 1};
    TreeNodeIndex upperNode = std::lower_bound(cbegin(tree), cend(tree), assignment.rangeEnd(myRank, 0)) - begin(tree);
    findHalos(tree, hNode, box, 0, upperNode, haloPairs);

    // group outgoing and incoming halo node indices by destination/source rank
    std::vector<std::vector<TreeNodeIndex>> incomingHaloNodes;
    std::vector<std::vector<TreeNodeIndex>> outgoingHaloNodes;
    computeSendRecvNodeList(tree, assignment, haloPairs, incomingHaloNodes, outgoingHaloNodes);

    // compute list of local node index ranges
    std::vector<TreeNodeIndex> incomingHalosFlattened = flattenNodeList(incomingHaloNodes);
    std::vector<TreeNodeIndex> localNodeRanges        = computeLocalNodeRanges(tree, assignment, myRank);

    std::vector<TreeNodeIndex> presentNodes;
    std::vector<TreeNodeIndex> nodeOffsets;
    computeLayoutOffsets(localNodeRanges, incomingHalosFlattened, counts, presentNodes, nodeOffsets);

    TreeNodeIndex firstLocalNode = std::lower_bound(cbegin(presentNodes), cend(presentNodes), localNodeRanges[0])
                         - begin(presentNodes);

    int newParticleStart = nodeOffsets[firstLocalNode];
    int newParticleEnd   = newParticleStart + assignment.totalCount(myRank);
    int nParticlesCore   = assignment.totalCount(myRank);
    int nParticlesExtracted = *nodeOffsets.rbegin();

    // (x,y,z)l arrays only contain the particles of the first assignment + halos
    std::vector<T> xl(nParticlesExtracted);
    std::vector<T> yl(nParticlesExtracted);
    std::vector<T> zl(nParticlesExtracted);
    std::vector<T> hl(nParticlesExtracted);
    std::vector<I> codesl(nParticlesExtracted);

    extractParticles(codes.data(), codes.data(), codes.size(), ordering.data(), presentNodes.data(), nodeOffsets.data(), presentNodes.size(),
                     tree.data(), codesl.data());

    extractParticles(x.data(), codes.data(), codes.size(), ordering.data(), presentNodes.data(), nodeOffsets.data(), presentNodes.size(),
                     tree.data(), xl.data());
    extractParticles(y.data(), codes.data(), codes.size(), ordering.data(), presentNodes.data(), nodeOffsets.data(), presentNodes.size(),
                     tree.data(), yl.data());
    extractParticles(z.data(), codes.data(), codes.size(), ordering.data(), presentNodes.data(), nodeOffsets.data(), presentNodes.size(),
                     tree.data(), zl.data());
    extractParticles(h.data(), codes.data(), codes.size(), ordering.data(), presentNodes.data(), nodeOffsets.data(), presentNodes.size(),
                     tree.data(), hl.data());

    int ngmax = 200;
    std::vector<int> neighbors(nParticles * ngmax);
    std::vector<int> neighborsCount(nParticles);
    std::vector<int> neighborsCore(nParticlesCore * ngmax);
    std::vector<int> neighborsCountCore(nParticlesCore);

    for (int i = newParticleStart; i < newParticleEnd; ++i)
    {
        I code = codesl[i];
        int iOrig = std::lower_bound(begin(codes), end(codes), code) - begin(codes);
        EXPECT_EQ(code, codes[iOrig]);

        // neighborsCount from reference (global) source data
        findNeighborsNaive(iOrig, x.data(), y.data(), z.data(), h.data(), nParticles, neighbors.data(),
                           neighborsCount.data(), ngmax, box);
        // neighborsCount from extracted particles for first assignment
        findNeighborsNaive(i, xl.data(), yl.data(), zl.data(), hl.data(), nParticlesExtracted, neighborsCore.data(),
                           neighborsCountCore.data(), ngmax, box);

        ASSERT_EQ(neighborsCountCore[i], neighborsCount[iOrig]);
        for (int ni = 0; ni < neighborsCountCore[i]; ++ni)
        {
            EXPECT_EQ(codesl[neighborsCore[i*ngmax + ni]], codes[neighbors[iOrig*ngmax+ni]]);
        }
    }
}

TEST(HaloNeighbors, randomGaussian)
{
    randomGaussianHaloNeighbors<unsigned, double>(false);
    randomGaussianHaloNeighbors<uint64_t, double>(false);
    randomGaussianHaloNeighbors<unsigned, float>(false);
    randomGaussianHaloNeighbors<uint64_t, float>(false);
}

TEST(HaloNeighbors, randomGaussianPbc)
{
    randomGaussianHaloNeighbors<unsigned, double>(true);
    randomGaussianHaloNeighbors<uint64_t, double>(true);
    randomGaussianHaloNeighbors<unsigned, float>(true);
    randomGaussianHaloNeighbors<uint64_t, float>(true);
}
