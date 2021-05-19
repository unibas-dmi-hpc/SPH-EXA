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

TEST(SfcCode, encodePlaceholderBit32)
{
    EXPECT_EQ(1,      encodePlaceholderBit(0u, 0));
    EXPECT_EQ(0b1000, encodePlaceholderBit(0u, 3));
    EXPECT_EQ(0b1010, encodePlaceholderBit(pad(0b010u, 3), 3));
    EXPECT_EQ(01635,  encodePlaceholderBit(06350000000u, 9));
}

TEST(SfcCode, decodePrefixLength32)
{
    EXPECT_EQ(0, decodePrefixLength(1u));
    EXPECT_EQ(3, decodePrefixLength(0b1000u));
    EXPECT_EQ(3, decodePrefixLength(0b1010u));
    EXPECT_EQ(9, decodePrefixLength(01635u));
}

TEST(SfcCode, decodePlaceholderbit32)
{
    EXPECT_EQ(0, decodePlaceholderBit(1u));
    EXPECT_EQ(0, decodePlaceholderBit(0b1000u));
    EXPECT_EQ(pad(0b010u, 3), decodePlaceholderBit(0b1010u));
    EXPECT_EQ(06350000000, decodePlaceholderBit(01635u));
}

TEST(SfcCode, encodePlaceholderBit64)
{
    EXPECT_EQ(1,      encodePlaceholderBit(0lu, 0));
    EXPECT_EQ(0b1000, encodePlaceholderBit(0lu, 3));
    EXPECT_EQ(0b1010, encodePlaceholderBit(pad(0b010lu, 3), 3));
    EXPECT_EQ(01635,  encodePlaceholderBit(0635000000000000000000ul, 9));
}

TEST(SfcCode, decodePrefixLength64)
{
    EXPECT_EQ(0, decodePrefixLength(1ul));
    EXPECT_EQ(3, decodePrefixLength(0b1000ul));
    EXPECT_EQ(3, decodePrefixLength(0b1010ul));
    EXPECT_EQ(9, decodePrefixLength(01635ul));
}

TEST(SfcCode, decodePlaceholderbit64)
{
    EXPECT_EQ(0, decodePlaceholderBit(1ul));
    EXPECT_EQ(0, decodePlaceholderBit(0b1000ul));
    EXPECT_EQ(pad(0b010ul, 3), decodePlaceholderBit(0b1010ul));
    EXPECT_EQ(0635000000000000000000ul, decodePlaceholderBit(01635ul));
}

TEST(SfcCode, octalDigit32)
{
    EXPECT_EQ(1, octalDigit(01234567012, 1));
    EXPECT_EQ(2, octalDigit(01234567012, 2));
    EXPECT_EQ(3, octalDigit(01234567012, 3));
    EXPECT_EQ(4, octalDigit(01234567012, 4));
    EXPECT_EQ(5, octalDigit(01234567012, 5));
    EXPECT_EQ(6, octalDigit(01234567012, 6));
    EXPECT_EQ(7, octalDigit(01234567012, 7));
    EXPECT_EQ(0, octalDigit(01234567012, 8));
    EXPECT_EQ(1, octalDigit(01234567012, 9));
    EXPECT_EQ(2, octalDigit(01234567012, 10));
}

TEST(SfcCode, octalDigit64)
{
    EXPECT_EQ(1, octalDigit(0123456701200000000000ul, 1));
    EXPECT_EQ(2, octalDigit(0123456701200000000000ul, 2));
    EXPECT_EQ(3, octalDigit(0123456701200000000000ul, 3));
    EXPECT_EQ(4, octalDigit(0123456701200000000000ul, 4));
    EXPECT_EQ(5, octalDigit(0123456701200000000000ul, 5));
    EXPECT_EQ(6, octalDigit(0123456701200000000000ul, 6));
    EXPECT_EQ(7, octalDigit(0123456701200000000000ul, 7));
    EXPECT_EQ(0, octalDigit(0123456701200000000000ul, 8));
    EXPECT_EQ(1, octalDigit(0123456701200000000000ul, 9));
    EXPECT_EQ(2, octalDigit(0123456701200000000000ul, 10));

    EXPECT_EQ(7, octalDigit(0123456701200000000007ul, 21));
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


/*! @brief return position of last non-zero octal digit place in x
 *
 * @tparam I   32- or 64-bit unsigned integer
 * @param  x   an integer
 * @return     position of the last non-zero octal digit place, starting from
 *             1 (left-most digit place) to 10 or 21 (64-bit), the right-most
 *             digit place. Returns 10 or 21 (64-bit) if x is zero.
 */
template<class I>
constexpr int lastNzPlace(I x)
{
    if (x) return maxTreeLevel<I>{} - __builtin_ctz(x)/3;
    else   return maxTreeLevel<I>{} - 0;
}

/*! @brief return the power of 8 for the octal place at position @p pos
 *
 * @tparam I    32- or 64-bit unsigned integer
 * @param  pos  Position counting from left, starting from 1. Maximum value 10 or 21 (64-bit)
 * @return      the power of 8 associated with the indicated octal place
 */
template<class I>
constexpr int octalPower(int pos)
{
    return (I(1) << 3 * (maxTreeLevel<I>{} - pos));
}

/*! @brief generate SFC codes to cover the range [a:b] with a valid cornerstone sub-octree
 *
 * @tparam I   32- or 64-bit unsigned integer
 * @param  a   first SFC code
 * @param  b   second SFC code
 *
 *                      | a_last_nz_pos (10)
 * Example:       a: 0001
 *                b: 0742
 *  ab_first_diff_pos ^ ^ b_last_nz_pos
 *       (8)                   (10)
 *
 *  output: 1 2 3 4 5 6 7 10 20 30 40 50 60 70 100 200 300 400 500 600 700 710 720 730 740 741
 *
 *  Variables suffixed with "_pos" refer to an octal digit place. The value of 1 is
 *  the position of the left-most digit, and 10 (or 21 for 64-bit) refers to the right-most digit.
 *  This convention is chosen such that the positional value coincides with the corresponding octree
 *  subdivision level.
 */
template<class I>
void spanSfcRange(I a, I b)
{
    // position of first differing octal digit place
    int ab_first_diff_pos = stl::min(1 + commonPrefix(a, b)/3, int(maxTreeLevel<I>{}));

    // last non-zero octal digit place position in a and b
    int a_last_nz_pos = lastNzPlace(a);
    int b_last_nz_pos = lastNzPlace(b);

    // add SFC codes, increasing power of 8 in each iteration
    for (int pos = a_last_nz_pos; pos > ab_first_diff_pos; --pos)
    {
        int numDigits = (8 - octalDigit(a, pos)) % 8;
        while (numDigits--)
        {
            a += octalPower<I>(pos);
            std::cout << std::oct << a << " ";
        }
    }

    // add SFC codes, decreasing power of 8 in each iteration
    for (int pos = ab_first_diff_pos; pos <= b_last_nz_pos; ++pos)
    {
        // Note: octalDigit(a, pos) is guaranteed zero for pos > ab_first_diff_pos
        int numDigits = octalDigit(b, pos) - octalDigit(a, pos);
        // suppress outputting b itself, the upper bound
        if (pos == b_last_nz_pos) { numDigits--; }
        while (numDigits--)
        {
            a += octalPower<I>(pos);
            std::cout << std::oct << a << " ";
        }
    }
}

TEST(SfcCode, spanSfcRange)
{
    spanSfcRange(0u, 07777u);
    std::cout << std::endl;
    spanSfcRange(1u, 0742u);
    std::cout << std::endl;
    spanSfcRange(040375u, 046023u);
    std::cout << std::endl;
    spanSfcRange(041305u, 046000u);
    std::cout << std::endl;
    spanSfcRange(040000u, 060000u);
    std::cout << std::endl;
}