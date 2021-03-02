/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! \file
 * \brief Test morton code implementation
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 */


#include <algorithm>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"
#include "cstone/sfc/mortoncode.hpp"
#include "cstone/sfc/mortonconversions.hpp"

using namespace cstone;

TEST(MortonCode, encode32) {

    double x = 0.00489; // 5 on the scale 0-1023
    double y = 0.00196; // 2
    double z = 0.00391; // 4

    // binary representation:
    // x = 101
    // y = 010
    // z = 100
    // Morton code is 101010100 = 340

    EXPECT_EQ(340, morton3DunitCube<unsigned>(float(x), float(y), float(z)));
    EXPECT_EQ(340, morton3DunitCube<unsigned>(x, y, z));
}

TEST(MortonCode, encode64) {

    double x = 0.5;
    double y = 0.5;
    double z = 0.5;

    // 21 bit inputs:
    // (2**20, 2**20, 2**20)
    // Morton code is box number 7 (=111) on the first split level, so
    // 0b0(111)(000)(000)...(000) = 0x7000000000000000lu

    std::size_t reference = 0x7000000000000000lu;
    EXPECT_EQ(reference, morton3DunitCube<std::size_t>(float(x), float(y), float(z)));
    EXPECT_EQ(reference, morton3DunitCube<std::size_t>(x, y, z));
}

template<class I>
void imorton3D()
{
    constexpr unsigned treeLevel = 3;
    std::array<unsigned, 3> box{ 5, 3, 6 };

    I testCode = imorton3D<I>(box[0], box[1], box[2], treeLevel);
    EXPECT_EQ(testCode, pad(I(0b101011110), 9));
}

TEST(MortonCode, imorton3D)
{
    imorton3D<unsigned>();
    imorton3D<uint64_t>();
}

TEST(MortonCode, decodeMorton32)
{
    unsigned x = 5;
    unsigned y = 2;
    unsigned z = 4;

    unsigned code = 340;
    EXPECT_EQ(x, decodeMortonX(code));
    EXPECT_EQ(y, decodeMortonY(code));
    EXPECT_EQ(z, decodeMortonZ(code));
}

TEST(MortonCode, decodeMorton64)
{
    std::size_t code = 0x7FFFFFFFFFFFFFFFlu;
    EXPECT_EQ((1u<<21u)-1u, decodeMortonX(code));
    EXPECT_EQ((1u<<21u)-1u, decodeMortonY(code));
    EXPECT_EQ((1u<<21u)-1u, decodeMortonZ(code));

    code = 0x1249249241249249;
    EXPECT_EQ((1u<<21u)-512u-1u, decodeMortonZ(code));

    code = 0b0111lu << (20u*3);
    EXPECT_EQ(1u<<20u, decodeMortonX(code));
    EXPECT_EQ(1u<<20u, decodeMortonY(code));
    EXPECT_EQ(1u<<20u, decodeMortonZ(code));

    code = 0b0011lu << (20u*3);
    EXPECT_EQ(0, decodeMortonX(code));
    EXPECT_EQ(1u<<20u, decodeMortonY(code));
    EXPECT_EQ(1u<<20u, decodeMortonZ(code));
}

TEST(MortonCode, decodeXRange32)
{
    EXPECT_EQ( (pair<int>{0,   512}),  decodeXRange(0b000u << 29u, 1));
    EXPECT_EQ( (pair<int>{512, 1024}), decodeXRange(0b001u << 29u, 1));

    EXPECT_EQ( (pair<int>{0,   512}),  decodeXRange(0b0000u << 28u, 2));
    EXPECT_EQ( (pair<int>{512, 1024}), decodeXRange(0b0010u << 28u, 2));

    EXPECT_EQ( (pair<int>{0,   512}),  decodeXRange(0b00000u << 27u, 3));
    EXPECT_EQ( (pair<int>{512, 1024}), decodeXRange(0b00100u << 27u, 3));

    EXPECT_EQ( (pair<int>{256,     256+256}),     decodeXRange(0b000001u << 26u, 4));
    EXPECT_EQ( (pair<int>{512+256, 512+256+256}), decodeXRange(0b001001u << 26u, 4));

    EXPECT_EQ( (pair<int>{512+256+128, 512+256+128+128}), decodeXRange(0b001001001u << 23u, 7));
    EXPECT_EQ( (pair<int>{512    +128, 512    +128+128}), decodeXRange(0b001000001u << 23u, 7));

    EXPECT_EQ( (pair<int>{512+256+128, 512+256+128+128}), decodeXRange(0b001101011u << 23u, 7));
    EXPECT_EQ( (pair<int>{512    +128, 512    +128+128}), decodeXRange(0b001100011u << 23u, 7));
}

TEST(MortonCode, decodeXRange64)
{
    EXPECT_EQ( (pair<int>{0,      1u<<20}), decodeXRange(0b00ul << 62u, 1));
    EXPECT_EQ( (pair<int>{1u<<20, 1u<<21}), decodeXRange(0b01ul << 62u, 1));

    EXPECT_EQ( (pair<int>{0,      1u<<20}), decodeXRange(0b000ul << 61u, 2));
    EXPECT_EQ( (pair<int>{1u<<20, 1u<<21}), decodeXRange(0b010ul << 61u, 2));

    EXPECT_EQ( (pair<int>{0,      1u<<20}), decodeXRange(0b0000ul << 60u, 3));
    EXPECT_EQ( (pair<int>{1u<<20, 1u<<21}), decodeXRange(0b0100ul << 60u, 3));

    EXPECT_EQ( (pair<int>{1u<<19,              1u<<20}), decodeXRange(0b00001ul << 59u, 4));
    EXPECT_EQ( (pair<int>{(1u<<20) + (1u<<19), 1u<<21}), decodeXRange(0b01001ul << 59u, 4));
}

TEST(MortonCode, decodeYRange32)
{
    EXPECT_EQ( (pair<int>{0,   512}),  decodeYRange(0b0000u << 28u, 2));
    EXPECT_EQ( (pair<int>{512, 1024}), decodeYRange(0b0001u << 28u, 2));

    EXPECT_EQ( (pair<int>{0,   512}),  decodeYRange(0b00000u << 27u, 3));
    EXPECT_EQ( (pair<int>{512, 1024}), decodeYRange(0b00010u << 27u, 3));

    EXPECT_EQ( (pair<int>{0,   512}),  decodeYRange(0b000000u << 26u, 4));
    EXPECT_EQ( (pair<int>{512, 1024}), decodeYRange(0b000100u << 26u, 4));

    EXPECT_EQ( (pair<int>{256,     256+256}),     decodeYRange(0b0000001u << 25u, 5));
    EXPECT_EQ( (pair<int>{512+256, 512+256+256}), decodeYRange(0b0001001u << 25u, 5));
}

TEST(MortonCode, decodeYRange64)
{
    EXPECT_EQ( (pair<int>{0,      1u<<20}), decodeYRange(0b000ul << 61u, 2));
    EXPECT_EQ( (pair<int>{1u<<20, 1u<<21}), decodeYRange(0b001ul << 61u, 2));

    EXPECT_EQ( (pair<int>{0,      1u<<20}), decodeYRange(0b0000ul << 60u, 3));
    EXPECT_EQ( (pair<int>{1u<<20, 1u<<21}), decodeYRange(0b0010ul << 60u, 3));

    EXPECT_EQ( (pair<int>{0,      1u<<20}), decodeYRange(0b00000ul << 59u, 4));
    EXPECT_EQ( (pair<int>{1u<<20, 1u<<21}), decodeYRange(0b00100ul << 59u, 4));

    EXPECT_EQ( (pair<int>{1u<<19,              1u<<20}), decodeYRange(0b000001ul << 58u, 5));
    EXPECT_EQ( (pair<int>{(1u<<20) + (1u<<19), 1u<<21}), decodeYRange(0b001001ul << 58u, 5));
}

TEST(MortonCode, decodeZRange32)
{
    EXPECT_EQ( (pair<int>{0,   512}),  decodeZRange(0b00000u << 27u, 3));
    EXPECT_EQ( (pair<int>{512, 1024}), decodeZRange(0b00001u << 27u, 3));

    EXPECT_EQ( (pair<int>{0,   512}),  decodeZRange(0b000000u << 26u, 4));
    EXPECT_EQ( (pair<int>{512, 1024}), decodeZRange(0b000010u << 26u, 4));

    EXPECT_EQ( (pair<int>{0,   512}),  decodeZRange(0b0000000u << 25u, 5));
    EXPECT_EQ( (pair<int>{512, 1024}), decodeZRange(0b0000100u << 25u, 5));

    EXPECT_EQ( (pair<int>{256,     256+256}),     decodeZRange(0b00000001u << 24u, 6));
    EXPECT_EQ( (pair<int>{512+256, 512+256+256}), decodeZRange(0b00001001u << 24u, 6));
}

TEST(MortonCode, decodeZRange64)
{
    EXPECT_EQ( (pair<int>{0,      1u<<20}), decodeZRange(0b0000ul << 60u, 3));
    EXPECT_EQ( (pair<int>{1u<<20, 1u<<21}), decodeZRange(0b0001ul << 60u, 3));

    EXPECT_EQ( (pair<int>{0,      1u<<20}), decodeZRange(0b00000ul << 59u, 4));
    EXPECT_EQ( (pair<int>{1u<<20, 1u<<21}), decodeZRange(0b00010ul << 59u, 4));

    EXPECT_EQ( (pair<int>{0,      1u<<20}), decodeZRange(0b000000ul << 58u, 5));
    EXPECT_EQ( (pair<int>{1u<<20, 1u<<21}), decodeZRange(0b000100ul << 58u, 5));

    EXPECT_EQ( (pair<int>{1u<<19,              1u<<20}), decodeZRange(0b0000001ul << 57u, 6));
    EXPECT_EQ( (pair<int>{(1u<<20) + (1u<<19), 1u<<21}), decodeZRange(0b0001001ul << 57u, 6));
}

TEST(MortonCode, codeFromIndices32)
{
    using CodeType = unsigned;

    constexpr unsigned maxLevel = maxTreeLevel<CodeType>{};

    std::array<unsigned char, 21> input{0};
    for (unsigned i = 0; i < maxLevel; ++i)
    {
        input[i] = 7;
    }

    EXPECT_EQ(nodeRange<CodeType>(0), codeFromIndices<CodeType>(input) + 1);
}

TEST(MortonCode, codeFromIndices64)
{
    using CodeType = uint64_t;

    constexpr unsigned maxLevel = maxTreeLevel<CodeType>{};

    std::array<unsigned char, 21> input{0};
    for (unsigned i = 0; i < maxLevel; ++i)
    {
        input[i] = 7;
    }

    EXPECT_EQ(nodeRange<CodeType>(0), codeFromIndices<CodeType>(input) + 1);
}

TEST(MortonCode, indicesFromCode32)
{
    using CodeType = unsigned;

    constexpr unsigned maxLevel = maxTreeLevel<CodeType>{};

    CodeType input = nodeRange<CodeType>(0);

    std::array<unsigned char, maxLevel> reference{0};
    for (unsigned i = 0; i < maxLevel; ++i)
    {
        reference[i] = 7;
    }

    EXPECT_EQ(reference, indicesFromCode(input - 1));

    input = 2 * nodeRange<CodeType>(3) + 4 * nodeRange<CodeType>(8);

    reference = std::array<unsigned char, maxLevel>{0,0,2,0,0,0,0,4,0,0};
    EXPECT_EQ(reference, indicesFromCode(input));
}

TEST(MortonCode, indicesFromCode64)
{
    using CodeType = uint64_t;

    constexpr unsigned maxLevel = maxTreeLevel<CodeType>{};

    CodeType input = nodeRange<CodeType>(0);

    std::array<unsigned char, maxLevel> reference{0};
    for (unsigned i = 0; i < maxLevel; ++i)
    {
        reference[i] = 7;
    }

    EXPECT_EQ(reference, indicesFromCode(input - 1));
}

template<class I>
void mortonNeighbors()
{
    //                      input    ref. out  treeLevel dx   dy   dz   PBC
    std::vector<std::tuple<  I,         I,     unsigned, int, int, int, bool>>
    codes{
        {pad(I(0b000111111), 9), pad(I(0b000111011), 9), 3, -1,  0,  0, false},
        {pad(I(0b000111111), 9), pad(I(0b100011011), 9), 3,  1,  0,  0, false},
        {pad(I(0b000111111), 9), pad(I(0b000111101), 9), 3,  0, -1,  0, false},
        {pad(I(0b000111111), 9), pad(I(0b010101101), 9), 3,  0,  1,  0, false},
        {pad(I(0b000111111), 9), pad(I(0b000111110), 9), 3,  0,  0, -1, false},
        {pad(I(0b000111111), 9), pad(I(0b001110110), 9), 3,  0,  0,  1, false},
        // over/underflow tests
        {pad(I(0b100111111), 9), pad(I(0b100111111), 9), 3,  1,  0,  0, false}, // overflow
        {pad(I(0b000011011), 9), pad(I(0b000011011), 9), 3, -1,  0,  0, false}, // underflow
        {pad(I(0b011), 3),       pad(I(0b111), 3),       1,  1,  0,  0, false},
        {pad(I(0b111), 3),       pad(I(0b111), 3),       1,  1,  0,  0, false}, // overflow
        {pad(I(0b011), 3),       pad(I(0b011), 3),       1, -1,  0,  0, false}, // underflow
        // diagonal offset
        {pad(I(0b000111111), 9), pad(I(0b000111000), 9), 3, -1, -1, -1, false},
        {pad(I(0b000111000), 9), pad(I(0b000111111), 9), 3,  1,  1,  1, false},
        // PBC cases
        {pad(I(0b000111111), 9), pad(I(0b000111011), 9), 3, -1,  0,  0, true},
        {pad(I(0b000111111), 9), pad(I(0b100011011), 9), 3,  1,  0,  0, true},
        {pad(I(0b000111111), 9), pad(I(0b000111101), 9), 3,  0, -1,  0, true},
        {pad(I(0b000111111), 9), pad(I(0b010101101), 9), 3,  0,  1,  0, true},
        {pad(I(0b000111111), 9), pad(I(0b000111110), 9), 3,  0,  0, -1, true},
        {pad(I(0b000111111), 9), pad(I(0b001110110), 9), 3,  0,  0,  1, true},
        // over/underflow tests
        {pad(I(0b100111111), 9), pad(I(0b000011011), 9), 3,  1,  0,  0, true}, // PBC sensitive
        {pad(I(0b000011011), 9), pad(I(0b100111111), 9), 3, -1,  0,  0, true}, // PBC sensitive
        {pad(I(0b011), 3),       pad(I(0b111), 3),       1,  1,  0,  0, true},
        {pad(I(0b111), 3),       pad(I(0b011), 3),       1,  1,  0,  0, true}, // PBC sensitive
        {pad(I(0b011), 3),       pad(I(0b111), 3),       1, -1,  0,  0, true}, // PBC sensitive
        // diagonal offset
        {pad(I(0b000111111), 9), pad(I(0b000111000), 9), 3, -1, -1, -1, true},
        {pad(I(0b000111000), 9), pad(I(0b000111111), 9), 3,  1,  1,  1, true},
        {pad(I(0b000111111), 9), pad(I(0b111111111), 9), 3, -4, -4, -4, true}, // PBC sensitive
        {pad(I(0b000111000), 9), pad(I(0b000000000), 9), 3,  6,  6,  6, true}, // PBC sensitive
    };

    auto computeCode = [](auto t)
    {
        bool usePbc = std::get<6>(t);
        return mortonNeighbor(std::get<0>(t), std::get<2>(t), std::get<3>(t),
                              std::get<4>(t), std::get<5>(t), usePbc, usePbc, usePbc);
    };

    std::vector<I> probes(codes.size());
    std::transform(begin(codes), end(codes), begin(probes), computeCode);

    for (std::size_t i = 0; i < codes.size(); ++i)
    {
        EXPECT_EQ(std::get<1>(codes[i]), probes[i]);
    }
}

TEST(MortonCode, mortonNeighbor32)
{
    mortonNeighbors<unsigned>();
    mortonNeighbors<uint64_t>();
}

TEST(MortonCode, mortonIndices32)
{
    using CodeType = unsigned;
    EXPECT_EQ(0x08000000, codeFromIndices<CodeType>({1}));
    EXPECT_EQ(0x09000000, codeFromIndices<CodeType>({1,1}));
    EXPECT_EQ(0x09E00000, codeFromIndices<CodeType>({1,1,7}));
}

TEST(MortonCode, mortonIndices64)
{
    using CodeType = uint64_t;
    EXPECT_EQ(0b0001lu << 60u, codeFromIndices<CodeType>({1}));
    EXPECT_EQ(0b0001001lu << 57u, codeFromIndices<CodeType>({1,1}));
    EXPECT_EQ(0b0001001111lu << 54u, codeFromIndices<CodeType>({1,1,7}));
}

TEST(MortonCode, mortonCodesSequence)
{
    constexpr double boxMin = -1;
    constexpr double boxMax = 1;
    Box<double> box(boxMin, boxMax);

    std::vector<double> x{-0.5, 0.5, -0.5, 0.5};
    std::vector<double> y{-0.5, 0.5, 0.5, -0.5};
    std::vector<double> z{-0.5, 0.5, 0.5, 0.5};

    std::vector<unsigned> reference;
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        reference.push_back(
            morton3DunitCube<unsigned>(normalize(x[i], boxMin, boxMax), normalize(y[i], boxMin, boxMax), normalize(z[i], boxMin, boxMax)));
    }

    std::vector<unsigned> probe(x.size());
    computeMortonCodes(begin(x), end(x), begin(y), begin(z), begin(probe), box);

    EXPECT_EQ(probe, reference);
}

