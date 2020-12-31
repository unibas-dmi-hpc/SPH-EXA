
#include "gtest/gtest.h"

#include "sfc/halos.hpp"

namespace sphexa
{

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
void findIncomingHalos()
{

    // a tree with 4 subdivisions along each dimension, 64 nodes
    std::vector<I> tree = makeUniformNLevelTree<I>(64, 1);

    // two particles per node
    std::vector<std::size_t> counts(nNodes(tree), 2);

    // two domains
    SpaceCurveAssignment<I> assignment(2);
    assignment.addRange(Rank(0), tree[0], tree[32], 64);
    assignment.addRange(Rank(1), tree[32], tree[64], 64);

    Box<double> box(0, 1);

    // size of one node is 0.25^3
    std::vector<double> interactionRadii(nNodes(tree), 0.1);

    std::vector<pair<int>> ranks{{0, 1}, {1, 0}};

    for (auto p : ranks)
    {
        int executingRank = p[0];
        int remoteRank    = p[1];
        std::vector<HaloRange<I>> halos
            = findIncomingHalos(tree, counts, interactionRadii, box, assignment, executingRank);

        std::vector<HaloRange<I>> refHalos;
        for (int y = 0; y < 4; ++y)
            for (int z = 0; z < 4; ++z)
            {
                int x = 2 - executingRank;
                I node = codeFromBox<I>(x, y, z, 2);
                refHalos.push_back({node, node + nodeRange<I>(2), 2, remoteRank});
            }

        std::sort(begin(refHalos), end(refHalos));

        EXPECT_EQ(halos, refHalos);
    }
}

} // namespace sphexa

TEST(Halos, findIncomingHalos)
{
    sphexa::findIncomingHalos<unsigned>();
    sphexa::findIncomingHalos<uint64_t>();
}
