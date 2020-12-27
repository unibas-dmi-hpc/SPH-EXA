
#include "gtest/gtest.h"

#include "sfc/collisions_cpu.hpp"
#include "sfc/octree_util.hpp"

#include "collision_reference/collisions_a2a.hpp"


using namespace sphexa;

/*! \brief compare tree-traversal collision detection with the naive all-to-all algorithm
 *
 * @tparam I           32- or 64-bit unsigned integer
 * @tparam T           float or double
 * @param tree         cornerstone octree leaves
 * @param haloRadii    floating point collision radius per octree leaf
 * @param box          bounding box used to construct the octree
 *
 * This test goes through all leaf nodes of the input octree and computes
 * a list of all other leaves that overlap with the first one.
 * The computation is done with both the tree-traversal algorithm and the
 * naive all-to-all algorithm and the results are compared.
 */
template<class I, class T>
void generalCollisionTest(const std::vector<I>& tree, const std::vector<T>& haloRadii,
                          const Box<T>& box)
{
    std::vector<BinaryNode<I>> internalTree = createInternalTree(tree);

    // tree traversal collision detection
    std::vector<CollisionList> collisions    = findAllCollisions(internalTree, tree, haloRadii, box);
    // naive all-to-all algorithm
    std::vector<CollisionList> refCollisions = findCollisionsAll2all(tree, haloRadii, box);

    for (int nodeIndex = 0; nodeIndex < nNodes(tree); ++nodeIndex)
    {
        std::vector<int> c{collisions[nodeIndex].begin(), collisions[nodeIndex].end()};
        std::vector<int> ref{refCollisions[nodeIndex].begin(), refCollisions[nodeIndex].end()};

        std::sort(begin(c), end(c));
        std::sort(begin(ref), end(ref));

        EXPECT_EQ(c, ref);
    }
}

//! \brief an irregular tree with level-3 nodes next to level-1 ones
template<class I, class T>
void irregularTreeTraversal()
{
    auto tree = OctreeMaker<I>{}.divide().divide(0).divide(0,7).makeTree();

    Box<T> box(0, 1);
    std::vector<T> haloRadii(nNodes(tree), 0.1);
    generalCollisionTest(tree, haloRadii, box);
}

TEST(Collisions, irregularTreeTraversal)
{
    irregularTreeTraversal<unsigned, float>();
    irregularTreeTraversal<uint64_t, float>();
    irregularTreeTraversal<unsigned, double>();
    irregularTreeTraversal<uint64_t, double>();
}


//! \brief an irregular tree with level-3 nodes next to level-1 ones
template<class I, class T>
void regularTreeTraversal()
{
    // 8x8x8 grid
    auto tree = makeUniformNLevelTree<I>(512, 1);

    Box<T> box(0, 1);
    // node edge length is 0.125
    std::vector<T> haloRadii(nNodes(tree), 0.124);
    generalCollisionTest(tree, haloRadii, box);
}

TEST(Collisions, regularTreeTraversal)
{
    regularTreeTraversal<unsigned, float>();
    regularTreeTraversal<uint64_t, float>();
    regularTreeTraversal<unsigned, double>();
    regularTreeTraversal<uint64_t, double>();
}

/*! \brief test tree traversal with anisotropic boxes
 *
 * anisotropic boxes with a single halo radius per node
 * results in different x,y,z halo search lengths once
 * the coordinates are normalized to the cubic unit box.
 */
class AnisotropicBoxTraversal : public testing::TestWithParam<std::array<int,6>>
{
public:
    template <class I, class T>
    void check()
    {
        // 8x8x8 grid
        auto tree = makeUniformNLevelTree<I>(512, 1);

        Box<T> box(std::get<0>(GetParam()),
                   std::get<1>(GetParam()),
                   std::get<2>(GetParam()),
                   std::get<3>(GetParam()),
                   std::get<4>(GetParam()),
                   std::get<5>(GetParam()));

        // node edge length is 0.125 in the compressed dimension
        // and 0.250 in the other two dimensions
        std::vector<T> haloRadii(nNodes(tree), 0.175);
        generalCollisionTest(tree, haloRadii, box);
    }
};

TEST_P(AnisotropicBoxTraversal, compressedAxis32f)
{
    check<unsigned, float>();
}

TEST_P(AnisotropicBoxTraversal, compressedAxis64f)
{
    check<uint64_t, float>();
}

TEST_P(AnisotropicBoxTraversal, compressedAxis32d)
{
    check<unsigned, double>();
}

TEST_P(AnisotropicBoxTraversal, compressedAxis64d)
{
    check<uint64_t, double>();
}

std::vector<std::array<int, 6>> boxLimits{{0,1,0,2,0,2},
                                          {0,2,0,1,0,2},
                                          {0,2,0,2,0,1}};

INSTANTIATE_TEST_SUITE_P(AnisotropicBoxTraversal,
                         AnisotropicBoxTraversal,
                         testing::ValuesIn(boxLimits));