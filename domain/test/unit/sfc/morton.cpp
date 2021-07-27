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

/*! @file
 * @brief Test morton code implementation
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <algorithm>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"
#include "cstone/sfc/morton.hpp"

using namespace cstone;

TEST(MortonCode, encode32)
{

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

TEST(MortonCode, encode64)
{

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

template<class KeyType>
void imorton3D()
{
    constexpr unsigned treeLevel = 3;
    std::array<unsigned, 3> box{5, 3, 6};

    KeyType testCode = imorton3D<KeyType>(box[0], box[1], box[2], treeLevel);
    EXPECT_EQ(testCode, pad(KeyType(0b101011110), 9));
}

TEST(MortonCode, imorton3D)
{
    imorton3D<unsigned>();
    imorton3D<uint64_t>();
}

TEST(MortonCode, idecodeMorton32)
{
    unsigned x = 5;
    unsigned y = 2;
    unsigned z = 4;

    unsigned code = 340;
    EXPECT_EQ(x, idecodeMortonX(code));
    EXPECT_EQ(y, idecodeMortonY(code));
    EXPECT_EQ(z, idecodeMortonZ(code));
}

TEST(MortonCode, idecodeMorton64)
{
    std::size_t code = 0x7FFFFFFFFFFFFFFFlu;
    EXPECT_EQ((1u << 21u) - 1u, idecodeMortonX(code));
    EXPECT_EQ((1u << 21u) - 1u, idecodeMortonY(code));
    EXPECT_EQ((1u << 21u) - 1u, idecodeMortonZ(code));

    code = 0x1249249241249249;
    EXPECT_EQ((1u << 21u) - 512u - 1u, idecodeMortonZ(code));

    code = 0b0111lu << (20u * 3);
    EXPECT_EQ(1u << 20u, idecodeMortonX(code));
    EXPECT_EQ(1u << 20u, idecodeMortonY(code));
    EXPECT_EQ(1u << 20u, idecodeMortonZ(code));

    code = 0b0011lu << (20u * 3);
    EXPECT_EQ(0, idecodeMortonX(code));
    EXPECT_EQ(1u << 20u, idecodeMortonY(code));
    EXPECT_EQ(1u << 20u, idecodeMortonZ(code));
}

template<class KeyType>
void decodeMortonXRange()
{
    Box<double> box(-1, 1);

    {
        KeyType prefix = pad(KeyType(012), 6); // (1,2)
        KeyType upper = pad(KeyType(013), 6);  // (1,3)

        auto x = decodeMortonXRange(prefix, upper, box);
        EXPECT_DOUBLE_EQ(x[0], -1.0);
        EXPECT_DOUBLE_EQ(x[1], -0.5);
    }
    {
        KeyType prefix = pad(KeyType(074), 6); // (7,4)
        KeyType upper = pad(KeyType(075), 6);  // (7,5)

        auto x = decodeMortonXRange(prefix, upper, box);
        EXPECT_DOUBLE_EQ(x[0], 0.5);
        EXPECT_DOUBLE_EQ(x[1], 1.0);
    }
}

TEST(MortonCode, decodeXRange)
{
    decodeMortonXRange<unsigned>();
    decodeMortonXRange<uint64_t>();
}

template<class KeyType>
void decodeMortonYRange()
{
    Box<double> box(-1, 1);

    {
        KeyType prefix = pad(KeyType(012), 6); // (1,2)
        KeyType upper = pad(KeyType(013), 6);  // (1,3)

        auto y = decodeMortonYRange(prefix, upper, box);
        EXPECT_DOUBLE_EQ(y[0], -0.5);
        EXPECT_DOUBLE_EQ(y[1], 0.0);
    }
    {
        KeyType prefix = pad(KeyType(074), 6); // (7,4)
        KeyType upper = pad(KeyType(075), 6);  // (7,5)

        auto y = decodeMortonYRange(prefix, upper, box);
        EXPECT_DOUBLE_EQ(y[0], 0.0);
        EXPECT_DOUBLE_EQ(y[1], 0.5);
    }
}

TEST(MortonCode, decodeYRange)
{
    decodeMortonYRange<unsigned>();
    decodeMortonYRange<uint64_t>();
}

template<class KeyType>
void decodeMortonZRange()
{
    Box<double> box(-1, 1);

    {
        KeyType prefix = pad(KeyType(012), 6); // (1,2)
        KeyType upper = pad(KeyType(013), 6);  // (1,3)

        auto z = decodeMortonZRange(prefix, upper, box);
        EXPECT_DOUBLE_EQ(z[0], 0.0);
        EXPECT_DOUBLE_EQ(z[1], 0.5);
    }
    {
        KeyType prefix = pad(KeyType(075), 6); // (7,5)
        KeyType upper = pad(KeyType(076), 6);  // (7,6)

        auto z = decodeMortonZRange(prefix, upper, box);
        EXPECT_DOUBLE_EQ(z[0], 0.5);
        EXPECT_DOUBLE_EQ(z[1], 1.0);
    }
}

TEST(MortonCode, decodeZRange)
{
    decodeMortonZRange<unsigned>();
    decodeMortonZRange<uint64_t>();
}

template<class KeyType>
void mortonNeighbors()
{
    //                      input    ref. out  treeLevel dx   dy   dz   PBC
    std::vector<std::tuple<KeyType, KeyType, unsigned, int, int, int, bool>> codes{
        {pad(KeyType(0b000111111), 9), pad(KeyType(0b000111011), 9), 3, -1, 0, 0, false},
        {pad(KeyType(0b000111111), 9), pad(KeyType(0b100011011), 9), 3, 1, 0, 0, false},
        {pad(KeyType(0b000111111), 9), pad(KeyType(0b000111101), 9), 3, 0, -1, 0, false},
        {pad(KeyType(0b000111111), 9), pad(KeyType(0b010101101), 9), 3, 0, 1, 0, false},
        {pad(KeyType(0b000111111), 9), pad(KeyType(0b000111110), 9), 3, 0, 0, -1, false},
        {pad(KeyType(0b000111111), 9), pad(KeyType(0b001110110), 9), 3, 0, 0, 1, false},
        // over/underflow tests
        {pad(KeyType(0b100111111), 9), pad(KeyType(0b100111111), 9), 3, 1, 0, 0, false},  // overflow
        {pad(KeyType(0b000011011), 9), pad(KeyType(0b000011011), 9), 3, -1, 0, 0, false}, // underflow
        {pad(KeyType(0b011), 3), pad(KeyType(0b111), 3), 1, 1, 0, 0, false},
        {pad(KeyType(0b111), 3), pad(KeyType(0b111), 3), 1, 1, 0, 0, false},  // overflow
        {pad(KeyType(0b011), 3), pad(KeyType(0b011), 3), 1, -1, 0, 0, false}, // underflow
        // diagonal offset
        {pad(KeyType(0b000111111), 9), pad(KeyType(0b000111000), 9), 3, -1, -1, -1, false},
        {pad(KeyType(0b000111000), 9), pad(KeyType(0b000111111), 9), 3, 1, 1, 1, false},
        // PBC cases
        {pad(KeyType(0b000111111), 9), pad(KeyType(0b000111011), 9), 3, -1, 0, 0, true},
        {pad(KeyType(0b000111111), 9), pad(KeyType(0b100011011), 9), 3, 1, 0, 0, true},
        {pad(KeyType(0b000111111), 9), pad(KeyType(0b000111101), 9), 3, 0, -1, 0, true},
        {pad(KeyType(0b000111111), 9), pad(KeyType(0b010101101), 9), 3, 0, 1, 0, true},
        {pad(KeyType(0b000111111), 9), pad(KeyType(0b000111110), 9), 3, 0, 0, -1, true},
        {pad(KeyType(0b000111111), 9), pad(KeyType(0b001110110), 9), 3, 0, 0, 1, true},
        // over/underflow tests
        {pad(KeyType(0b100111111), 9), pad(KeyType(0b000011011), 9), 3, 1, 0, 0, true},  // PBC sensitive
        {pad(KeyType(0b000011011), 9), pad(KeyType(0b100111111), 9), 3, -1, 0, 0, true}, // PBC sensitive
        {pad(KeyType(0b011), 3), pad(KeyType(0b111), 3), 1, 1, 0, 0, true},
        {pad(KeyType(0b111), 3), pad(KeyType(0b011), 3), 1, 1, 0, 0, true},  // PBC sensitive
        {pad(KeyType(0b011), 3), pad(KeyType(0b111), 3), 1, -1, 0, 0, true}, // PBC sensitive
        // diagonal offset
        {pad(KeyType(0b000111111), 9), pad(KeyType(0b000111000), 9), 3, -1, -1, -1, true},
        {pad(KeyType(0b000111000), 9), pad(KeyType(0b000111111), 9), 3, 1, 1, 1, true},
        {pad(KeyType(0b000111111), 9), pad(KeyType(0b111111111), 9), 3, -4, -4, -4, true}, // PBC sensitive
        {pad(KeyType(0b000111000), 9), pad(KeyType(0b000000000), 9), 3, 6, 6, 6, true},    // PBC sensitive
    };

    auto computeCode = [](auto t) {
        bool usePbc = std::get<6>(t);
        return mortonNeighbor(std::get<0>(t), std::get<2>(t), std::get<3>(t), std::get<4>(t), std::get<5>(t), usePbc,
                              usePbc, usePbc);
    };

    std::vector<KeyType> probes(codes.size());
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
