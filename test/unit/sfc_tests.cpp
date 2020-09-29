#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "sfc/zorder.hpp"


TEST(SFC, mortonIndex) {

    double x = 0.00489; // 5 on the scale 0-1023
    double y = 0.00196; // 2
    double z = 0.00391; // 4

    // binary representation:
    // x = 101
    // y = 010
    // z = 100
    // Morton code is 101010100 = 340

    EXPECT_EQ(340, sphexa::morton3D(float(x), float(y), float(z)));
    EXPECT_EQ(340, sphexa::morton3D(x, y, z));
}

TEST(SFC, sortInvert)
{
    std::vector<int> v{2,1,5,4};

    // the sort keys that sorts v is be {1,0,3,2}
    std::vector<int> sortKey(v.size());

    sphexa::sort_invert(begin(v), end(v), begin(sortKey));

    
    std::vector<int> reference{1,0,3,2};
    EXPECT_EQ(sortKey, reference);
}

TEST(SFC, mortonCodes)
{
    using sphexa::detail::normalize;

    constexpr double boxMin = -1;
    constexpr double boxMax = 1;

    std::vector<double> x{-0.5, 0.5, -0.5, 0.5};
    std::vector<double> y{-0.5, 0.5, 0.5, -0.5};
    std::vector<double> z{-0.5, 0.5, 0.5, 0.5};

    std::vector<unsigned int> reference;
    for (int i = 0; i < x.size(); ++i)
    {
        reference.push_back(
            sphexa::morton3D(normalize(x[i], boxMin, boxMax),
                             normalize(y[i], boxMin, boxMax),
                             normalize(z[i], boxMin, boxMax))
        );
    }

    std::vector<unsigned> probe(x.size());
    sphexa::computeMortonCodes(begin(x), end(x), begin(y), begin(z), begin(probe),
                               boxMin, boxMax, boxMin, boxMax, boxMin, boxMax);

    EXPECT_EQ(probe, reference);
}

TEST(SFC, computeZorder)
{
    // assume BBox of [-1, 1]^3
    constexpr double boxMin = -1;
    constexpr double boxMax = 1;

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
    sphexa::computeZorder(begin(x), end(x), begin(y), begin(z), begin(zOrder),
                          boxMin, boxMax, boxMin, boxMax, boxMin, boxMax);

    EXPECT_EQ(zOrder, reference);
}

int main(int argc, char **argv) {

  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
