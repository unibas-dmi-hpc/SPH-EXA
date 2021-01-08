
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


TEST(Collisions, adjacentEdgeRegression)
{
    std::vector<unsigned> tree{0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32, 40, 48, 56, 64, 128, 192, 256, 320, 384,
                               448, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 8192, 12288, 16384, 20480, 24576,
                               28672, 32768, 65536, 98304, 131072, 163840, 196608, 229376, 262144, 524288, 786432, 1048576,
                               1310720, 1572864, 1835008, 2097152, 4194304, 6291456, 8388608, 10485760, 12582912, 14680064,
                               16777216, 33554432, 50331648, 67108864, 83886080, 100663296, 117440512, 134217728, 268435456,
                               402653184, 536870912, 671088640, 805306368, 939524096, 956301312, 973078528, 989855744, 1006632960,
                               1023410176, 1040187392, 1056964608, 1059061760, 1061158912, 1063256064, 1065353216, 1067450368,
                               1069547520, 1071644672, 1071906816, 1072168960, 1072431104, 1072693248, 1072955392, 1073217536,
                               1073479680, 1073512448, 1073545216, 1073577984, 1073610752, 1073643520, 1073676288, 1073709056,
                               1073713152, 1073717248, 1073721344, 1073725440, 1073729536, 1073733632, 1073737728, 1073738240,
                               1073738752, 1073739264, 1073739776, 1073740288, 1073740800, 1073741312, 1073741376, 1073741440,
                               1073741504, 1073741568, 1073741632, 1073741696, 1073741760, 1073741768, 1073741776, 1073741784,
                               1073741792, 1073741800, 1073741808, 1073741816, 1073741817, 1073741818, 1073741819, 1073741820,
                               1073741821, 1073741822, 1073741823, 1073741824};

    auto internalTree = createInternalTree(tree);

    Box<double> box(0.5, 0.6);
    //Box<double> box(0, 1);

    std::vector<double> haloRadii(nNodes(tree), 0);
    haloRadii[0] = 0.2;
    *haloRadii.rbegin() = 0.2;

    int lastNode = nNodes(tree) - 1;
    Box<int> haloBox = makeHaloBox(tree[lastNode], tree[lastNode+1], haloRadii[lastNode], box);

    //std::cout << haloBox.xmin() << " "  << haloBox.xmax() << std::endl;
    //std::cout << haloBox.ymin() << " "  << haloBox.ymax() << std::endl;
    //std::cout << haloBox.zmin() << " "  << haloBox.zmax() << std::endl;

    CollisionList collisions;
    findCollisions(internalTree.data(), tree.data(), collisions, haloBox);

    std::vector<int> cnodes{collisions.begin(), collisions.end()};
    std::sort(begin(cnodes), end(cnodes));

    //for (auto node : cnodes)
    //    std::cout << node << " ";
    //std::cout << std::endl;

    //EXPECT_EQ(collisions.size(), nNodes(tree));
    //generalCollisionTest(tree, haloRadii, box);
}