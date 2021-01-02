
#include "gtest/gtest.h"

#include "sfc/halos.hpp"

using namespace sphexa;

/*! \brief find halo test
 *
 * A regular 4x4x4 tree with 64 nodes is assigned to 2 ranks,
 * such that nodes 0-32 are on rank 0 and 32-64 on rank 1,
 * or, in x,y,z coordinates,
 *
 * nodes (0-2, 0-4, 0-4) -> rank 0
 * nodes (2-4, 0-4, 0-4) -> rank 1
 *
 * Halo search radius is less then a node edge length, so the halo nodes are
 *
 * (2, 0-4, 0-4) halos of rank 0
 * (1, 0-4, 0-4) halos of rank 1
 */
template <class I>
void findHalos()
{
    // a tree with 4 subdivisions along each dimension, 64 nodes
    std::vector<I> tree = makeUniformNLevelTree<I>(64, 1);

    // two domains
    SpaceCurveAssignment<I> assignment(2);
    assignment.addRange(Rank(0), tree[0], tree[32], 64);
    assignment.addRange(Rank(1), tree[32], tree[64], 64);

    Box<double> box(0, 1);

    // size of one node is 0.25^3
    std::vector<double> interactionRadii(nNodes(tree), 0.1);

    std::vector<pair<int>> refPairs0;
    for (int i = 0; i < nNodes(tree) / 2; ++i)
        for (int j = nNodes(tree) / 2; j < nNodes(tree); ++j)
        {
            if (overlap(tree[i], tree[i + 1], makeHaloBox(tree[j], tree[j + 1], interactionRadii[j], box)))
            {
                refPairs0.emplace_back(i, j);
            }
        }
    std::sort(begin(refPairs0), end(refPairs0));
    EXPECT_EQ(refPairs0.size(), 100);

    {
        std::vector<pair<int>> testPairs0;
        findHalos(tree, interactionRadii, box, assignment, 0, testPairs0);
        std::sort(begin(testPairs0), end(testPairs0));

        EXPECT_EQ(testPairs0.size(), 100);
        EXPECT_EQ(testPairs0, refPairs0);
    }

    auto refPairs1 = refPairs0;
    for (auto& p : refPairs1)
        std::swap(p[0], p[1]);
    std::sort(begin(refPairs1), end(refPairs1));

    {
        std::vector<pair<int>> testPairs1;
        findHalos(tree, interactionRadii, box, assignment, 1, testPairs1);
        std::sort(begin(testPairs1), end(testPairs1));
        EXPECT_EQ(testPairs1.size(), 100);
        EXPECT_EQ(testPairs1, refPairs1);
    }
}

TEST(Halos, findHalos)
{
    findHalos<unsigned>();
    findHalos<uint64_t>();
}


//! \brief test processing of halo pair nodes into send/receive node lists
template <class I>
void prepareHaloExchange()
{
    // a tree with 4 subdivisions along each dimension, 64 nodes
    std::vector<I> tree = makeUniformNLevelTree<I>(64, 1);

    // two domains
    SpaceCurveAssignment<I> assignment(2);
    assignment.addRange(Rank(0), tree[0], tree[32], 64);
    assignment.addRange(Rank(1), tree[32], tree[64], 64);

    Box<double> box(0, 1);

    // size of one node is 0.25^3
    std::vector<double> interactionRadii(nNodes(tree), 0.1);

    std::vector<pair<int>> haloPairs;
    for (int i = 0; i < nNodes(tree) / 2; ++i)
        for (int j = nNodes(tree) / 2; j < nNodes(tree); ++j)
        {
            if (overlap(tree[i], tree[i + 1], makeHaloBox(tree[j], tree[j + 1], interactionRadii[j], box)))
            {
                haloPairs.emplace_back(i, j);
            }
        }
    std::sort(begin(haloPairs), end(haloPairs));

    {
        std::vector<std::vector<int>> incomingHalos(assignment.nRanks());
        std::vector<std::vector<int>> outgoingHalos(assignment.nRanks());

        prepareHaloExchange<I>(tree, assignment, haloPairs, incomingHalos, outgoingHalos);

        std::vector<std::vector<int>> refIncomingHalos(assignment.nRanks());
        std::vector<std::vector<int>> refOutgoingHalos(assignment.nRanks());

        std::vector<int> frontier0{4,5,6,7,12,13,14,15,20,21,22,23,28,29,30,31};
        std::vector<int> frontier1{32,33,34,35,40,41,42,43,48,49,50,51,56,57,58,59};

        refIncomingHalos[1] = frontier1;
        refOutgoingHalos[1] = frontier0;

        EXPECT_EQ(incomingHalos, refIncomingHalos);
        EXPECT_EQ(outgoingHalos, refOutgoingHalos);
    }
}

TEST(Halos, prepareHaloExchange)
{
    prepareHaloExchange<unsigned>();
    prepareHaloExchange<uint64_t>();
}