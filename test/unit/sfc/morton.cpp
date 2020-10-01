#include <algorithm>
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

TEST(SFC, decodeMorton32)
{
    unsigned x = 5;
    unsigned y = 2;
    unsigned z = 4;

    unsigned code = 340;
    EXPECT_EQ(x, sphexa::decodeMortonX(code));
    EXPECT_EQ(y, sphexa::decodeMortonY(code));
    EXPECT_EQ(z, sphexa::decodeMortonZ(code));
}

TEST(SFC, decodeMorton64)
{
    std::size_t code = 0x7FFFFFFFFFFFFFFFlu;
    EXPECT_EQ((1u<<21u)-1u, sphexa::decodeMortonX(code));
    EXPECT_EQ((1u<<21u)-1u, sphexa::decodeMortonY(code));
    EXPECT_EQ((1u<<21u)-1u, sphexa::decodeMortonZ(code));

    code = 0x1249249241249249;
    EXPECT_EQ((1u<<21u)-512u-1u, sphexa::decodeMortonZ(code));

    code = 0b0111lu << (20u*3);
    EXPECT_EQ(1u<<20u, sphexa::decodeMortonX(code));
    EXPECT_EQ(1u<<20u, sphexa::decodeMortonY(code));
    EXPECT_EQ(1u<<20u, sphexa::decodeMortonZ(code));

    code = 0b0011lu << (20u*3);
    EXPECT_EQ(0, sphexa::decodeMortonX(code));
    EXPECT_EQ(1u<<20u, sphexa::decodeMortonY(code));
    EXPECT_EQ(1u<<20u, sphexa::decodeMortonZ(code));
}

TEST(SFC, enclosingBox)
{
    std::size_t code      = 0x0FF0000000000001;
    std::size_t reference = 0x0FC0000000000000;
    EXPECT_EQ(reference, sphexa::detail::enclosingBoxCode(code, 3));

    unsigned code_u = 0x07F00001;
    unsigned reference_u = 0x07E00000;
    EXPECT_EQ(reference_u, sphexa::detail::enclosingBoxCode(code_u, 3));
}

TEST(SFC, mortonNeighbor)
{
    {
        unsigned code = 0x07E00000;
        // move one node up in x-direction at tree level 3
        unsigned xUpCode = sphexa::mortonNeighbor(code, 3, 1, 0, 0);
        unsigned reference = 0x23600000;

        EXPECT_EQ(reference, xUpCode);
    }
    {
        std::vector<std::size_t> codes{
            0b0000111111lu << (18u*3),
            0b0000111111lu << (18u*3),
            0b0000111111lu << (18u*3),
            0b0000111111lu << (18u*3),
            0b0000111111lu << (18u*3),
            0b0000111111lu << (18u*3),
            0b0011lu << (20u*3),
            0b0111lu << (20u*3),
        };
        std::vector<std::size_t> references{
            0b0000111011lu << (18u*3),
            0b0100011011lu << (18u*3),
            0b0000111101lu << (18u*3),
            0b0010101101lu << (18u*3),
            0b0000111110lu << (18u*3),
            0b0001110110lu << (18u*3),
            0b0111lu << (20u*3),
            0b0111lu << (20u*3),
        };
        std::vector<std::size_t> probes{
            sphexa::mortonNeighbor(codes[0], 3, -1, 0, 0),
            sphexa::mortonNeighbor(codes[1], 3, 1, 0, 0),
            sphexa::mortonNeighbor(codes[2], 3, 0, -1, 0),
            sphexa::mortonNeighbor(codes[3], 3, 0, 1, 0),
            sphexa::mortonNeighbor(codes[4], 3, 0, 0, -1),
            sphexa::mortonNeighbor(codes[5], 3, 0, 0, 1),
            sphexa::mortonNeighbor(codes[6], 1, 1, 0, 0),
            sphexa::mortonNeighbor(codes[7], 1, 1, 0, 0),
        };

        // move one node up in x-direction at tree level 3
        //std::size_t xUpCode = sphexa::mortonNeighbor(code, 3, 1, 0, 0);
        EXPECT_EQ(references, probes);
    }
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

