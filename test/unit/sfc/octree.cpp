
#include "gtest/gtest.h"

#include "sfc/mortoncode.hpp"
#include "sfc/octree.hpp"

using sphexa::detail::codeFromBox;

template <class I>
class ExampleOctree
{
public:
    ExampleOctree()
        : codes{
        // 000
           // 000 000
           codeFromBox<I>({0,0,1}, 2),
           codeFromBox<I>({0,1,0}, 2),
           codeFromBox<I>({0,1,1}, 2),
           codeFromBox<I>({1,0,0}, 2),
           codeFromBox<I>({1,0,1}, 2),
           codeFromBox<I>({1,1,0}, 2),
           codeFromBox<I>({1,1,1}, 2),
        // 001
           codeFromBox<I>({0,0,1}, 1),
        // 010
           codeFromBox<I>({0,1,0}, 1),
        // 011
           codeFromBox<I>({0,1,1}, 1),
        // 100
           codeFromBox<I>({1,0,0}, 1),
        // 101
           codeFromBox<I>({1,0,1}, 1),
        // 110
           codeFromBox<I>({1,1,0}, 1),
        // 111
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
