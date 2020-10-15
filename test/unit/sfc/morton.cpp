#include <algorithm>
#include <iostream>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"
#include "sfc/mortoncode.hpp"

TEST(MortonCode, mortonIndex32) {

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

TEST(MortonCode, mortonIndex64) {

    double x = 0.5;
    double y = 0.5;
    double z = 0.5;

    // 21 bit inputs:
    // (2**20, 2**20, 2**20)
    // Morton code is box number 7 (=111) on the first split level, so
    // 0b0(111)(000)(000)...(000) = 0x7000000000000000lu

    std::size_t reference = 0x7000000000000000lu;
    EXPECT_EQ(reference, sphexa::morton3D<std::size_t>(float(x), float(y), float(z)));
    EXPECT_EQ(reference, sphexa::morton3D<std::size_t>(x, y, z));
}

TEST(MortonCode, decodeMorton32)
{
    unsigned x = 5;
    unsigned y = 2;
    unsigned z = 4;

    unsigned code = 340;
    EXPECT_EQ(x, sphexa::decodeMortonX(code));
    EXPECT_EQ(y, sphexa::decodeMortonY(code));
    EXPECT_EQ(z, sphexa::decodeMortonZ(code));
}

TEST(MortonCode, decodeMorton64)
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

TEST(MortonCode, enclosingBoxTrim)
{
    std::size_t code      = 0x0FF0000000000001;
    std::size_t reference = 0x0FC0000000000000;
    EXPECT_EQ(reference, sphexa::detail::enclosingBoxCode(code, 3));

    unsigned code_u = 0x07F00001;
    unsigned reference_u = 0x07E00000;
    EXPECT_EQ(reference_u, sphexa::detail::enclosingBoxCode(code_u, 3));
}

TEST(MortonCode, enclosingBoxMaxLevel32)
{
    using CodeType = unsigned;
    CodeType code  = 0x0FF00001;
    CodeType probe = sphexa::detail::enclosingBoxCode(code, sphexa::maxTreeLevel<CodeType>{});
    EXPECT_EQ(probe, code);
}

TEST(MortonCode, enclosingBoxMaxLevel64)
{
    using CodeType = uint64_t;
    CodeType code  = 0x0FF0000000000001;
    CodeType probe = sphexa::detail::enclosingBoxCode(code, sphexa::maxTreeLevel<CodeType>{});
    EXPECT_EQ(probe, code);
}

TEST(MortonCode, smallestCommonBoxEqualCode)
{
    using CodeType = unsigned;
    CodeType code = 0;
    auto probe = sphexa::smallestCommonBox(code, code);
    std::tuple<CodeType, CodeType> reference{code, code + 1};
    EXPECT_EQ(probe, reference);
}

TEST(MortonCode, smallestCommonBoxL1)
{
    using CodeType = unsigned;
    CodeType code1 = 0b00001001u << 24u;
    CodeType code2 = 0b00001010u << 24u;
    auto probe = sphexa::smallestCommonBox(code1, code2);
    std::tuple<CodeType, CodeType> reference{0b00001000u<<24u, 0b000010000u << 24u};
    EXPECT_EQ(probe, reference);
}

TEST(MortonCode, smallestCommonBoxL0_32)
{
    using CodeType = unsigned;
    CodeType code1 = 0b00000001u << 24u;
    CodeType code2 = 0b00001010u << 24u;
    auto probe = sphexa::smallestCommonBox(code1, code2);
    std::tuple<CodeType, CodeType> reference{0u, 0b01u << 30u};
    EXPECT_EQ(probe, reference);
}

TEST(MortonCode, smallestCommonBoxL0_64)
{
    using CodeType = uint64_t;
    CodeType code1 = 0b0000001lu << 57u;
    CodeType code2 = 0b0001010lu << 57u;
    auto probe = sphexa::smallestCommonBox(code1, code2);
    std::tuple<CodeType, CodeType> reference{0lu, 1lu << 63u};
    EXPECT_EQ(probe, reference);
}

TEST(MortonCode, boxCoordinates32)
{
    constexpr unsigned treeLevel = 3;
    // (5,3,6)
    unsigned code = 0b00101011110u << (7u*3);

    auto c = sphexa::detail::boxFromCode(code, treeLevel);

    std::array<unsigned, 3> cref{ 5, 3, 6 };
    EXPECT_EQ(c, cref);
}

TEST(MortonCode, boxCoordinates64)
{
    constexpr unsigned treeLevel = 3;
    // (5,3,6)
    uint64_t inputCode = 0b0101011110ul << (18u*3);

    auto c = sphexa::detail::boxFromCode(inputCode, treeLevel);

    std::array<unsigned, 3> cref{ 5, 3, 6 };
    EXPECT_EQ(c, cref);
}

TEST(MortonCode, codeFromBox32)
{
    using CodeType = unsigned;

    constexpr unsigned treeLevel = 3;
    std::array<unsigned, 3> box{ 5, 3, 6 };

    CodeType testCode = sphexa::detail::codeFromBox<CodeType>(box, treeLevel);

    std::array<unsigned, 3> testBox = sphexa::detail::boxFromCode(testCode, treeLevel);
    EXPECT_EQ(testBox, box);
}

TEST(MortonCode, codeFromBox64)
{
    using CodeType = uint64_t;

    constexpr unsigned treeLevel = 3;
    std::array<unsigned, 3> box{ 5, 3, 6 };

    CodeType testCode = sphexa::detail::codeFromBox<CodeType>(box, treeLevel);

    std::array<unsigned, 3> testBox = sphexa::detail::boxFromCode(testCode, treeLevel);
    EXPECT_EQ(testBox, box);
}

TEST(MortonCode, mortonNeighbor32)
{
    std::vector<std::tuple<unsigned, unsigned, unsigned, int, int, int>> codes{
        {0b00000111111u << (7u*3), 0b00000111011u << (7u*3), 3, -1,  0,  0},
        {0b00000111111u << (7u*3), 0b00100011011u << (7u*3), 3,  1,  0,  0},
        {0b00000111111u << (7u*3), 0b00000111101u << (7u*3), 3,  0, -1,  0},
        {0b00000111111u << (7u*3), 0b00010101101u << (7u*3), 3,  0,  1,  0},
        {0b00000111111u << (7u*3), 0b00000111110u << (7u*3), 3,  0,  0, -1},
        {0b00000111111u << (7u*3), 0b00001110110u << (7u*3), 3,  0,  0,  1},
        // over/underflow tests
        {0b00100111111u << (7u*3), 0b00100111111u << (7u*3), 3,  1,  0,  0}, // overflow
        {0b00000011011u << (7u*3), 0b00000011011u << (7u*3), 3, -1,  0,  0}, // underflow
        {0b00011u << (9u*3),       0b00111lu << (9u*3),       1,  1,  0,  0},
        {0b00111u << (9u*3),       0b00111lu << (9u*3),       1,  1,  0,  0}, // overflow
        {0b00011u << (9u*3),       0b00011lu << (9u*3),       1, -1,  0,  0}, // underflow
        // diagonal offset
        {0b00000111111u << (7u*3), 0b00111000u << (7u*3), 3, -1, -1, -1},
        {0b00000111000u << (7u*3), 0b00111111u << (7u*3), 3,  1,  1,  1},
    };

    auto computeCode = [](auto t)
    {
      return sphexa::mortonNeighbor(std::get<0>(t), std::get<2>(t), std::get<3>(t),
                                    std::get<4>(t), std::get<5>(t));
    };

    std::vector<unsigned> probes(codes.size());
    std::transform(begin(codes), end(codes), begin(probes), computeCode);

    for (int i = 0; i < codes.size(); ++i)
    {
        EXPECT_EQ(std::get<1>(codes[i]), probes[i]);
    }
}

TEST(MortonCode, mortonNeighbor64)
{
    std::vector<std::tuple<std::size_t, std::size_t, unsigned, int, int, int>> codes{
        {0b0000111111lu << (18u*3), 0b0000111011lu << (18u*3), 3, -1,  0,  0},
        {0b0000111111lu << (18u*3), 0b0100011011lu << (18u*3), 3,  1,  0,  0},
        {0b0000111111lu << (18u*3), 0b0000111101lu << (18u*3), 3,  0, -1,  0},
        {0b0000111111lu << (18u*3), 0b0010101101lu << (18u*3), 3,  0,  1,  0},
        {0b0000111111lu << (18u*3), 0b0000111110lu << (18u*3), 3,  0,  0, -1},
        {0b0000111111lu << (18u*3), 0b0001110110lu << (18u*3), 3,  0,  0,  1},
        // over/underflow tests
        {0b0100111111lu << (18u*3), 0b0100111111lu << (18u*3), 3,  1,  0,  0}, // overflow
        {0b0000011011lu << (18u*3), 0b0000011011lu << (18u*3), 3, -1,  0,  0}, // underflow
        {0b0011lu << (20u*3),       0b0111lu << (20u*3),       1,  1,  0,  0},
        {0b0111lu << (20u*3),       0b0111lu << (20u*3),       1,  1,  0,  0}, // overflow
        {0b0011lu << (20u*3),       0b0011lu << (20u*3),       1, -1,  0,  0}, // underflow
        // diagonal offset
        {0b0000111111lu << (18u*3), 0b0111000lu << (18u*3), 3, -1, -1, -1},
        {0b0000111000lu << (18u*3), 0b0111111lu << (18u*3), 3,  1,  1,  1},
    };

    auto computeCode = [](auto t)
    {
        return sphexa::mortonNeighbor(std::get<0>(t), std::get<2>(t), std::get<3>(t),
                                      std::get<4>(t), std::get<5>(t));
    };

    std::vector<std::size_t> probes(codes.size());
    std::transform(begin(codes), end(codes), begin(probes), computeCode);

    for (int i = 0; i < codes.size(); ++i)
    {
        EXPECT_EQ(std::get<1>(codes[i]), probes[i]);
    }
}

TEST(MortonCode, mortonIndices32)
{
    using CodeType = unsigned;
    EXPECT_EQ(0x08000000, sphexa::detail::codeFromIndices<CodeType>({1}));
    EXPECT_EQ(0x09000000, sphexa::detail::codeFromIndices<CodeType>({1,1}));
    EXPECT_EQ(0x09E00000, sphexa::detail::codeFromIndices<CodeType>({1,1,7}));
}

TEST(MortonCode, mortonIndices64)
{
    using CodeType = uint64_t;
    EXPECT_EQ(0b0001lu << 60u, sphexa::detail::codeFromIndices<CodeType>({1}));
    EXPECT_EQ(0b0001001lu << 57u, sphexa::detail::codeFromIndices<CodeType>({1,1}));
    EXPECT_EQ(0b0001001111lu << 54u, sphexa::detail::codeFromIndices<CodeType>({1,1,7}));
}

TEST(MortonCode, mortonCodesSequence)
{
    using sphexa::detail::normalize;

    constexpr double boxMin = -1;
    constexpr double boxMax = 1;
    sphexa::Box<double> box(boxMin, boxMax);

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
    sphexa::computeMortonCodes(begin(x), end(x), begin(y), begin(z), begin(probe), box);

    EXPECT_EQ(probe, reference);
}

