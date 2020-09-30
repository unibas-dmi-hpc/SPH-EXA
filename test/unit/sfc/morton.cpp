#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "sfc/mortoncode.hpp"

TEST(SFC, mortonIndex32) {

    double x = 0.00489; // 5 on the scale 0-1023
    double y = 0.00196; // 2
    double z = 0.00391; // 4

    // binary representation:
    // x = 101
    // y = 010
    // z = 100
    // Morton code is 101010100 = 340

    EXPECT_EQ(340, sphexa::morton3D<unsigned>(float(x), float(y), float(z)));
    EXPECT_EQ(340, sphexa::morton3D<unsigned>(x, y, z));
}

TEST(SFC, mortonIndex64) {

    double x = 0.5;
    double y = 0.5;
    double z = 0.5;

    // 21 bit inputs:
    // (2**20, 2**20, 2**20)
    // Morton code is box number 7 (=111) on the first split level, so
    // 0x0(111)(000)(000)...(000) = 0x7000000000000000lu

    std::size_t reference = 0x7000000000000000lu;
    EXPECT_EQ(reference, sphexa::morton3D<std::size_t>(float(x), float(y), float(z)));
    EXPECT_EQ(reference, sphexa::morton3D<std::size_t>(x, y, z));
}

TEST(SFC, mortonIndices)
{
    EXPECT_EQ(0x08000000, sphexa::mortonFromIndices({1}));
    EXPECT_EQ(0x09000000, sphexa::mortonFromIndices({1,1}));
    EXPECT_EQ(0x09E00000, sphexa::mortonFromIndices({1,1,7}));
}

TEST(SFC, mortonCodes)
{
    using sphexa::detail::normalize;

    constexpr double boxMin = -1;
    constexpr double boxMax = 1;

    std::vector<double> x{-0.5, 0.5, -0.5, 0.5};
    std::vector<double> y{-0.5, 0.5, 0.5, -0.5};
    std::vector<double> z{-0.5, 0.5, 0.5, 0.5};

    std::vector<unsigned> reference;
    for (int i = 0; i < x.size(); ++i)
    {
        reference.push_back(
            sphexa::morton3D<unsigned>(normalize(x[i], boxMin, boxMax), normalize(y[i], boxMin, boxMax), normalize(z[i], boxMin, boxMax)));
    }

    std::vector<unsigned> probe(x.size());
    sphexa::computeMortonCodes(begin(x), end(x), begin(y), begin(z), begin(probe), boxMin, boxMax, boxMin, boxMax, boxMin, boxMax);

    EXPECT_EQ(probe, reference);
}

