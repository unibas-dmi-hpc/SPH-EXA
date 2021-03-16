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
 * @brief Test common SFC functionality
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */


#include "gtest/gtest.h"
#include "cstone/sfc/common.hpp"

using namespace cstone;

template<class T>
void normalization32()
{
    EXPECT_EQ(toNBitInt<unsigned>(T(0)), 0);
    EXPECT_EQ(toNBitInt<unsigned>(T(0.0000001)), 0);
    EXPECT_EQ(toNBitInt<unsigned>(T(1)), 1023);

    EXPECT_EQ(toNBitInt<unsigned>(T(0.00489)), 5);
    EXPECT_EQ(toNBitInt<unsigned>(T(0.00196)), 2);
    EXPECT_EQ(toNBitInt<unsigned>(T(0.00391)), 4);
}

TEST(SfcCode, normalization32)
{
    normalization32<float>();
    normalization32<double>();
}

template<class T>
void normalization64()
{
    EXPECT_EQ(toNBitInt<uint64_t>(T(0)), 0);
    EXPECT_EQ(toNBitInt<unsigned>(T(0.0000001)), 0);
    EXPECT_EQ(toNBitInt<uint64_t>(T(1)), 2097151);

    EXPECT_EQ(toNBitInt<uint64_t>(T(0.00489)), 10255);
    EXPECT_EQ(toNBitInt<uint64_t>(T(0.00196)), 4110);
    EXPECT_EQ(toNBitInt<uint64_t>(T(0.00391)), 8199);
}

TEST(SfcCode, normalization64)
{
    normalization64<float>();
    normalization64<double>();
}

template<class T>
void normalizationCeil32()
{
    EXPECT_EQ(toNBitIntCeil<unsigned>(T(0)), 0);
    EXPECT_EQ(toNBitIntCeil<unsigned>(T(0.0000001)), 1);
    EXPECT_EQ(toNBitIntCeil<unsigned>(T(1)), 1023);

    EXPECT_EQ(toNBitIntCeil<unsigned>(T(0.00489)), 6);
    EXPECT_EQ(toNBitIntCeil<unsigned>(T(0.00196)), 3);
    EXPECT_EQ(toNBitIntCeil<unsigned>(T(0.00391)), 5);
}

TEST(SfcCode, normalizationCeil32)
{
    normalizationCeil32<float>();
    normalizationCeil32<double>();
}

template<class T>
void normalizationCeil64()
{
    EXPECT_EQ(toNBitIntCeil<uint64_t>(T(0)), 0);
    EXPECT_EQ(toNBitIntCeil<unsigned>(T(0.0000001)), 1);
    EXPECT_EQ(toNBitIntCeil<uint64_t>(T(1)), 2097151);

    EXPECT_EQ(toNBitIntCeil<uint64_t>(T(0.00489)), 10256);
    EXPECT_EQ(toNBitIntCeil<uint64_t>(T(0.00196)), 4111);
    EXPECT_EQ(toNBitIntCeil<uint64_t>(T(0.00391)), 8200);
}

TEST(SfcCode, normalizationCeil64)
{
    normalizationCeil64<float>();
    normalizationCeil64<double>();
}

TEST(SfcCode, zeroLowBits32)
{
    EXPECT_EQ( (0b00111000u << 24u), zeroLowBits( (0b00111111u << 24u), 3));
    EXPECT_EQ( (0b00110000u << 24u), zeroLowBits( (0b00111111u << 24u), 2));
    EXPECT_EQ( (0b00100000u << 24u), zeroLowBits( (0b00111111u << 24u), 1));
}

TEST(SfcCode, zeroLowBits64)
{
    EXPECT_EQ( (0b0111000lu << 57u), zeroLowBits( (0b0111111lu << 57u), 3));
    EXPECT_EQ( (0b0110000lu << 57u), zeroLowBits( (0b0111111lu << 57u), 2));
    EXPECT_EQ( (0b0100000lu << 57u), zeroLowBits( (0b0111111lu << 57u), 1));
}

TEST(SfcCode, log8ceil32)
{
    EXPECT_EQ(2, log8ceil(64u));
    EXPECT_EQ(3, log8ceil(100u));
}

TEST(SfcCode, log8ceil64)
{
    EXPECT_EQ(2, log8ceil(64lu));
    EXPECT_EQ(3, log8ceil(100lu));
}

TEST(SfcCode, treeLevel32)
{
    using CodeType = unsigned;
    EXPECT_EQ(0, treeLevel<CodeType>(1u<<30u));
    EXPECT_EQ(1, treeLevel<CodeType>(1u<<27u));
    EXPECT_EQ(10, treeLevel<CodeType>(1));
}

TEST(SfcCode, treeLevel64)
{
    using CodeType = uint64_t;
    EXPECT_EQ(0, treeLevel<CodeType>(1ul<<63u));
    EXPECT_EQ(1, treeLevel<CodeType>(1ul<<60u));
    EXPECT_EQ(21, treeLevel<CodeType>(1));
}

TEST(SfcCode, parentIndex32)
{
    EXPECT_EQ(6, parentIndex(6u, 10));
    EXPECT_EQ(1, parentIndex(1u<<24u, 2));
    EXPECT_EQ(2, parentIndex(1u<<25u, 2));
    EXPECT_EQ(1, parentIndex(1u<<27u, 1));
}

TEST(SfcCode, parentIndex64)
{
    EXPECT_EQ(6, parentIndex(6ul, 21));
    EXPECT_EQ(1, parentIndex(1ul<<57u, 2));
    EXPECT_EQ(2, parentIndex(1ul<<58u, 2));
    EXPECT_EQ(1, parentIndex(1ul<<60u, 1));
}

TEST(SfcCode, enclosingBoxTrim)
{
    std::size_t code      = 0x0FF0000000000001;
    std::size_t reference = 0x0FC0000000000000;
    EXPECT_EQ(reference, enclosingBoxCode(code, 3));

    unsigned code_u = 0x07F00001;
    unsigned reference_u = 0x07E00000;
    EXPECT_EQ(reference_u, enclosingBoxCode(code_u, 3));
}

TEST(SfcCode, enclosingBoxMaxLevel32)
{
    using CodeType = unsigned;
    CodeType code  = 0x0FF00001;
    CodeType probe = enclosingBoxCode(code, maxTreeLevel<CodeType>{});
    EXPECT_EQ(probe, code);
}

TEST(SfcCode, enclosingBoxMaxLevel64)
{
    using CodeType = uint64_t;
    CodeType code  = 0x0FF0000000000001;
    CodeType probe = enclosingBoxCode(code, maxTreeLevel<CodeType>{});
    EXPECT_EQ(probe, code);
}

TEST(SfcCode, smallestCommonBoxEqualCode)
{
    using CodeType = unsigned;
    CodeType code = 0;
    auto probe = smallestCommonBox(code, code);
    pair<CodeType> reference{code, code + 1};
    EXPECT_EQ(probe, reference);
}

TEST(SfcCode, smallestCommonBoxL1)
{
    using CodeType = unsigned;
    CodeType code1 = 0b00001001u << 24u;
    CodeType code2 = 0b00001010u << 24u;
    auto probe = smallestCommonBox(code1, code2);
    pair<CodeType> reference{0b00001000u<<24u, 0b000010000u << 24u};
    EXPECT_EQ(probe, reference);
}

TEST(SfcCode, smallestCommonBoxL0_32)
{
    using CodeType = unsigned;
    CodeType code1 = 0b00000001u << 24u;
    CodeType code2 = 0b00001010u << 24u;
    auto probe = smallestCommonBox(code1, code2);
    pair<CodeType> reference{0u, 0b01u << 30u};
    EXPECT_EQ(probe, reference);
}

TEST(SfcCode, smallestCommonBoxL0_64)
{
    using CodeType = uint64_t;
    CodeType code1 = 0b0000001lu << 57u;
    CodeType code2 = 0b0001010lu << 57u;
    auto probe = smallestCommonBox(code1, code2);
    pair<CodeType> reference{0lu, 1lu << 63u};
    EXPECT_EQ(probe, reference);
}

TEST(SfcCode, padUtility)
{
    EXPECT_EQ(pad(0b011,   3), 0b00011 << 27);
    EXPECT_EQ(pad(0b011ul, 3), 0b0011ul << 60);
}

