#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "sfc/mortoncode.hpp"

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


