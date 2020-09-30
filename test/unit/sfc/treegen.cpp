#include <vector>

#include "gtest/gtest.h"

#include "sfc/treegen.hpp"

TEST(SFC, generateTree)
{
    std::vector<unsigned> mortonCodes{0,1,2,3,4,5,6,7};

    EXPECT_NO_THROW(sphexa::generateOctree(mortonCodes, 0, 0, 0, 0, 0, 0));
}
