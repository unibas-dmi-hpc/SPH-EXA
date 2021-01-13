

#include "gtest/gtest.h"

#include "sfc/findneighbors.hpp"
#include "sfc/halodiscovery.hpp"
#include "sfc/layout.hpp"

#include "coord_samples/random.hpp"

/*! \brief \file Combines halo search with neighbor search
 *
 * We test that halo search finds all particles required for a complete
 * neighbor search.
 *
 */

using namespace sphexa;

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


template<class I, class T>
void randomGaussianHaloNeighbors()
{
    int nParticles = 1000;
    int bucketSize = 10;
    T   smoothingL = 0.1;

    int nRanks     = 2;

    Box<T> box{-1, 1};
    RandomCoordinates<T, I> coords(nParticles, box);

    std::vector<I> codes = coords.mortonCodes();

    std::vector<T> x = coords.x();
    std::vector<T> y = coords.y();
    std::vector<T> z = coords.z();
    std::vector<T> h(nParticles, smoothingL);

    std::vector<int> ordering(nParticles);
    std::iota(begin(ordering), end(ordering), 0);

    auto [tree, counts] =
        computeOctree(coords.mortonCodes().data(), coords.mortonCodes().data() + nParticles,
                                    bucketSize);

    std::vector<T> hNode(nNodes(tree));
    computeNodeMax(tree.data(), nNodes(tree), codes.data(), codes.data() + nParticles, ordering.data(),
                   h.data(), hNode.data());

    auto assignment = singleRangeSfcSplit(tree, counts, nRanks);

    // find halos for rank 0
    int myRank = 0;
    std::vector<pair<int>> haloPairs;
    findHalos(tree, hNode, box, assignment, myRank, haloPairs);

    // group outgoing and incoming halo node indices by destination/source rank
    std::vector<std::vector<int>> incomingHaloNodes;
    std::vector<std::vector<int>> outgoingHaloNodes;
    computeSendRecvNodeList(tree, assignment, haloPairs, incomingHaloNodes, outgoingHaloNodes);

    // compute list of local node index ranges
    std::vector<int> incomingHalosFlattened = flattenNodeList(incomingHaloNodes);
    std::vector<int> localNodeRanges        = computeLocalNodeRanges(tree, assignment, myRank);

    std::vector<int> presentNodes;
    std::vector<int> nodeOffsets;
    computeLayoutOffsets(localNodeRanges, incomingHalosFlattened, counts, presentNodes, nodeOffsets);
}

TEST(HaloNeighbors, randomGaussian)
{
    randomGaussianHaloNeighbors<unsigned, double>();
}