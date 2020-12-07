#include<vector>

#include "gtest/gtest.h"

#include "sfc/halos.hpp"

namespace sphexa
{

template <class I>
void findIncomingHalos()
{

    // a tree with 4 subdivisions along each dimension, 64 nodes
    std::vector<I> tree = detail::makeUniformNLevelTree<I>(64, 1);

    // two particles per node
    std::vector<std::size_t> counts(nNodes(tree), 2);

    // two domains
    SpaceCurveAssignment<I> assignment(2);
    assignment[0].addRange(tree[0], tree[32], 64);
    assignment[1].addRange(tree[32], tree[64], 64);

    Box<double> box(-1, 1);

    // size of one node is 0.5^3
    std::vector<double> interactionRadii(nNodes(tree), 0.25);

    auto halos = findIncomingHalos(tree, counts, interactionRadii, box, assignment, 0);
}

} // namespace sphexa

TEST(Halos, findIncomingHalos)
{
    sphexa::findIncomingHalos<unsigned>();
}
