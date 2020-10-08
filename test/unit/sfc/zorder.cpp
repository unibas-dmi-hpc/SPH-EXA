#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "sfc/zorder.hpp"


TEST(SFC, sortInvert)
{
    std::vector<int> v{2,1,5,4};

    // the sort keys that sorts v is {1,0,3,2}
    std::vector<int> sortKey(v.size());

    sphexa::sort_invert(begin(v), end(v), begin(sortKey));

    
    std::vector<int> reference{1,0,3,2};
    EXPECT_EQ(sortKey, reference);
}

TEST(SFC, computeZorder)
{
    // assume BBox of [-1, 1]^3
    constexpr double boxMin = -1;
    constexpr double boxMax = 1;
    sphexa::Box<double> box{boxMin, boxMax};

    // 8 particles, each centered in each of the 8 octants,
    // Z-indices            4    5      1     6     3     0     2    7
    // position             0    1      2     3     4     5     6    7
    std::vector<double> x{ 0.5,  0.5, -0.5,  0.5, -0.5, -0.5, -0.5, 0.5};
    std::vector<double> y{-0.5, -0.5, -0.5,  0.5,  0.5, -0.5,  0.5, 0.5};
    std::vector<double> z{-0.5,  0.5,  0.5, -0.5,  0.5, -0.5, -0.5, 0.5};

    // the sort order to access coordinates in ascending Z-order is
    // 5, 2, 6, 4, 0, 1, 3, 7
    std::vector<unsigned> reference{5,2,6,4,0,1,3,7};

    std::vector<unsigned> zOrder(x.size());
    sphexa::computeZorder(begin(x), end(x), begin(y), begin(z), begin(zOrder), box);

    EXPECT_EQ(zOrder, reference);
}

