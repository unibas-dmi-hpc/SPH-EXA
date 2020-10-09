
#include "gtest/gtest.h"

#include "sfc/mortoncode.hpp"
#include "sfc/octree.hpp"

template<class I>
class ExampleOctree
{
public:
    ExampleOctree()
    {
        I code = 0;
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
