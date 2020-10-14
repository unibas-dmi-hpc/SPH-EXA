
#include "gtest/gtest.h"

#include "sfc/mortoncode.hpp"
#include "sfc/octree.hpp"

using sphexa::detail::codeFromIndices;

template <class I>
class ExampleOctree
{
public:
    ExampleOctree()
        : codes{
        // 000
           codeFromIndices<I>({0,1}),
           codeFromIndices<I>({0,2}),
           codeFromIndices<I>({0,3}),
           codeFromIndices<I>({0,4}),
           codeFromIndices<I>({0,5}),
           codeFromIndices<I>({0,6}),
           codeFromIndices<I>({0,7}),
        // 001
           codeFromIndices<I>({1}),
        // 010
           codeFromIndices<I>({2,1,0}),
           codeFromIndices<I>({2,1,1}),
           codeFromIndices<I>({2,1,2}),
           codeFromIndices<I>({2,1,3}),
           codeFromIndices<I>({2,1,4}),
           codeFromIndices<I>({2,1,5}),
           codeFromIndices<I>({2,1,6}),
           codeFromIndices<I>({2,1,7}),
        // 011
           codeFromIndices<I>({3}),
        // 100
           codeFromIndices<I>({4}),
        // 101
           codeFromIndices<I>({5}),
        // 110
           codeFromIndices<I>({6}),
        // 111
           codeFromIndices<I>({7, 0}),
           codeFromIndices<I>({7, 0}) + 1,
           codeFromIndices<I>({7, 1}),
           codeFromIndices<I>({7, 2}),
           codeFromIndices<I>({7, 3}),
           codeFromIndices<I>({7, 4}),
           codeFromIndices<I>({7, 5}),
           codeFromIndices<I>({7, 6, 0}),
           codeFromIndices<I>({7, 6, 1}),
           codeFromIndices<I>({7, 6, 2}),
           codeFromIndices<I>({7, 6, 3}),
           codeFromIndices<I>({7, 6, 4}),
           codeFromIndices<I>({7, 6, 5}),
           codeFromIndices<I>({7, 6, 6}),
           codeFromIndices<I>({7, 7}),
        },
        nodes{
            sphexa::SfcNode<I>{codeFromIndices<I>({0}), codeFromIndices<I>({1}), 0, 7},
            sphexa::SfcNode<I>{codeFromIndices<I>({1}), codeFromIndices<I>({2}), 7, 1},
            sphexa::SfcNode<I>{codeFromIndices<I>({2,1,0}), codeFromIndices<I>({2,1,1}), 8, 1},
            sphexa::SfcNode<I>{codeFromIndices<I>({2,1,1}), codeFromIndices<I>({2,1,2}), 9, 1},
            sphexa::SfcNode<I>{codeFromIndices<I>({2,1,2}), codeFromIndices<I>({2,1,3}), 10, 1},
            sphexa::SfcNode<I>{codeFromIndices<I>({2,1,3}), codeFromIndices<I>({2,1,4}), 11, 1},
            sphexa::SfcNode<I>{codeFromIndices<I>({2,1,4}), codeFromIndices<I>({2,1,5}), 12, 1},
            sphexa::SfcNode<I>{codeFromIndices<I>({2,1,5}), codeFromIndices<I>({2,1,6}), 13, 1},
            sphexa::SfcNode<I>{codeFromIndices<I>({2,1,6}), codeFromIndices<I>({2,1,7}), 14, 1},
            sphexa::SfcNode<I>{codeFromIndices<I>({2,1,7}), codeFromIndices<I>({2,2}), 15, 1},
            sphexa::SfcNode<I>{codeFromIndices<I>({3}), codeFromIndices<I>({4}), 16, 1},
            sphexa::SfcNode<I>{codeFromIndices<I>({4}), codeFromIndices<I>({5}), 17, 1},
            sphexa::SfcNode<I>{codeFromIndices<I>({5}), codeFromIndices<I>({6}), 18, 1},
            sphexa::SfcNode<I>{codeFromIndices<I>({6}), codeFromIndices<I>({7}), 19, 1},

            sphexa::SfcNode<I>{codeFromIndices<I>({7,0}), codeFromIndices<I>({7,1}), 20, 2},
            sphexa::SfcNode<I>{codeFromIndices<I>({7,1}), codeFromIndices<I>({7,2}), 22, 1},
            sphexa::SfcNode<I>{codeFromIndices<I>({7,2}), codeFromIndices<I>({7,3}), 23, 1},
            sphexa::SfcNode<I>{codeFromIndices<I>({7,3}), codeFromIndices<I>({7,4}), 24, 1},
            sphexa::SfcNode<I>{codeFromIndices<I>({7,4}), codeFromIndices<I>({7,5}), 25, 1},
            sphexa::SfcNode<I>{codeFromIndices<I>({7,5}), codeFromIndices<I>({7,6}), 25, 1},
            sphexa::SfcNode<I>{codeFromIndices<I>({7,6}), codeFromIndices<I>({7,7}), 26, 7},
            sphexa::SfcNode<I>{codeFromIndices<I>({7,7}), 8*codeFromIndices<I>({1}), 33, 1},
        }
    {
    }

private:
    std::vector<I> codes;
    // expected resulting tree
    std::vector<sphexa::SfcNode<I>> nodes;
};

TEST(Octree, construct)
{
    unsigned bucketSize = 2;

    std::vector<unsigned> codes;

    sphexa::generateOctree(codes, bucketSize);
}
