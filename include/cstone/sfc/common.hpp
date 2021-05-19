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
 * @brief  Common operations on SFC keys that do not depend on the specific SFC used
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <cassert>
#include <cstdint>
#include <type_traits>

#include "cstone/primitives/clz.hpp"
#include "cstone/primitives/stl.hpp"
#include "cstone/util/util.hpp"

namespace cstone
{
//! @brief number of unused leading zeros in a 32-bit SFC code
template<class I>
struct unusedBits : stl::integral_constant<unsigned, 2> {};

//! @brief number of unused leading zeros in a 64-bit SFC code
template<>
struct unusedBits<uint64_t> : stl::integral_constant<unsigned, 1> {};

template<class I>
struct maxTreeLevel : stl::integral_constant<unsigned, 10> {};

template<>
struct maxTreeLevel<uint64_t> : stl::integral_constant<unsigned, 21> {};


/*! @brief normalize a floating point number in [0,1] to an integer in [0, 2^(10 or 21)-1]
 *
 * @tparam I  32-bit or 64-bit unsigned integer
 * @tparam T  float or double
 * @param x   input floating point number in [0,1]
 * @return    x converted to an 10-bit or 21-bit integer
 *
 * Integer conversion happens with truncation as required for morton code calculations
 */
template <class I, class T>
CUDA_HOST_DEVICE_FUN
inline unsigned toNBitInt(T x)
{
    // spatial resolution in bits per dimension
    constexpr unsigned nBits = maxTreeLevel<I>{};

    // [0,1] to [0,1023] and convert to integer (32-bit) or
    // [0,1] to [0,2097151] and convert to integer (64-bit)
    return stl::min(stl::max(x * T(1u<<nBits), T(0.0)), T((1u<<nBits)-1u));
}

/*! @brief normalize a floating point number in [0,1] to an integer in [0, 2^(10 or 21)-1]
 *
 * @tparam I  32-bit or 64-bit unsigned integer
 * @tparam T  float or double
 * @param x   input floating point number in [0,1]
 * @return    x converted to an 10-bit or 21-bit integer
 *
 * Integer conversion happens with ceil() as required for converting halo radii to integers
 * where we must round up to the smallest integer not less than x*2^(10 or 21)
 */
template <class I, class T>
CUDA_HOST_DEVICE_FUN
constexpr unsigned toNBitIntCeil(T x)
{
    // spatial resolution in bits per dimension
    constexpr unsigned nBits = maxTreeLevel<I>{};

    // [0,1] to [0,1023] and convert to integer (32-bit) or
    // [0,1] to [0,2097151] and convert to integer (64-bit)
    return stl::min(stl::max(std::ceil(x * T(1u<<nBits)), T(0.0)), T((1u<<nBits)-1u));
}

/*! @brief add (binary) zeros behind a prefix
 *
 * Allows comparisons, such as number of leading common bits (cpr)
 * of the prefix with SFC codes.
 *
 * @tparam I      32- or 64-bit unsigned integer type
 * @param prefix  the bit pattern
 * @param length  number of bits in the prefix
 * @return        prefix padded out with zeros
 *
 * Examples:
 *  pad(0b011u,  3) -> 0b00011 << 27
 *  pad(0b011ul, 3) -> 0b0011ul << 60
 *
 *  i.e. @p length plus the number of zeros added adds up to 30 for 32-bit integers
 *  or 63 for 64-bit integers, because these are the numbers of usable bits in SFC codes.
 */
template <class I>
constexpr I pad(I prefix, int length)
{
    return prefix << (3*maxTreeLevel<I>{} - length);
}

/*! @brief compute the maximum range of an octree node at a given subdivision level
 *
 * @tparam I         32- or 64-bit unsigned integer type
 * @param treeLevel  octree subdivision level
 * @return           the range
 *
 * At treeLevel 0, the range is the entire 30 or 63 bits used in the SFC code.
 * After that, the range decreases by 3 bits for each level.
 *
 */
template<class I>
CUDA_HOST_DEVICE_FUN
constexpr std::enable_if_t<std::is_unsigned<I>{}, I> nodeRange(unsigned treeLevel)
{
    assert (treeLevel <= maxTreeLevel<I>{});
    unsigned shifts = maxTreeLevel<I>{} - treeLevel;

    I ret = I(1) << (3u * shifts);
    return ret;
}

//! @brief compute ceil(log8(n))
template<class I>
CUDA_HOST_DEVICE_FUN
constexpr std::enable_if_t<std::is_unsigned<I>{}, unsigned> log8ceil(I n)
{
    if (n == 0) { return 0; }

    unsigned lz = countLeadingZeros(n-1);
    return maxTreeLevel<I>{} - (lz - unusedBits<I>{}) / 3;
}

//! @brief check whether n is a power of 8
template<class I>
CUDA_HOST_DEVICE_FUN
constexpr std::enable_if_t<std::is_unsigned<I>{}, bool> isPowerOf8(I n)
{
    unsigned lz = countLeadingZeros(n - 1) - unusedBits<I>{};
    return lz % 3 == 0 && !(n & (n-1));
}

/*! @brief calculate common prefix (cpr) of two morton keys
 *
 * @tparam I    32 or 64 bit unsigned integer
 * @param key1  first morton code key
 * @param key2  second morton code key
 * @return      number of continuous identical bits, counting from MSB
 *              minus the 2 unused bits in 32 bit codes or minus the 1 unused bit
 *              in 64 bit codes.
 */
template<class I>
CUDA_HOST_DEVICE_FUN
constexpr int commonPrefix(I key1, I key2)
{
    return int(countLeadingZeros(key1 ^ key2)) - unusedBits<I>{};
}

/*! @brief return octree subdivision level corresponding to codeRange
 *
 * @tparam I         32- or 64-bit unsigned integer type
 * @param codeRange  input SFC code range
 * @return           octree subdivision level 0-10 (32-bit) or 0-21 (64-bit)
 */
template<class I>
CUDA_HOST_DEVICE_FUN
constexpr unsigned treeLevel(I codeRange)
{
    assert( isPowerOf8(codeRange) );
    return (countLeadingZeros(codeRange - 1) - unusedBits<I>{}) / 3;
}

/*! @brief convert a plain SFC key into the placeholder bit format (Warren-Salmon 1993)
 *
 * @tparam I               32- or 64-bit unsigned integer
 * @param code             input SFC key
 * @param prefixLength     number of leading bits which are part of the code
 * @return                 code shifted by trailing zeros and prepended with 1-bit
 *
 * Example: encodePlaceholderBit(06350000000, 9) -> 01635 (in octal)
 */
template<class I>
CUDA_HOST_DEVICE_FUN
constexpr I encodePlaceholderBit(I code, int prefixLength)
{
    int nShifts = 3*maxTreeLevel<I>{} - prefixLength;
    I ret = code >> nShifts;
    I placeHolderMask = I(1) << prefixLength;

    return placeHolderMask | ret;
}

//! @brief returns the number of key-bits in the input @p code
template<class I>
CUDA_HOST_DEVICE_FUN
constexpr int decodePrefixLength(I code)
{
    return 8*sizeof(I) - 1 - countLeadingZeros(code);
}

/*! @brief decode an SFC key in Warren-Salmon placeholder bit format
 *
 * @tparam I         32- or 64-bit unsigned integer
 * @param code       input SFC key with 1-bit prepended
 * @return           SFC-key without 1-bit and shifted to most significant bit
 *
 * Inverts encodePlaceholderBit.
 */
template<class I>
CUDA_HOST_DEVICE_FUN
constexpr I decodePlaceholderBit(I code)
{
    int prefixLength  = decodePrefixLength(code);
    I placeHolderMask = I(1) << prefixLength;
    I ret = code ^ placeHolderMask;

    return ret << (3*maxTreeLevel<I>{} - prefixLength);
}

/*! @brief extract the n-th octal digit from an SFC key, starting from the most significant
 *
 * @tparam I         32- or 64-bit unsigned integer type
 * @param code       Input SFC key code
 * @param position   Which digit place to extract. Return values will be meaningful for
 *                   @p position in [1:11] for 32-bit keys and in [1:22] for 64-bit keys and
 *                   will be zero otherwise, but a value of 0 for @p position can also be specified
 *                   to detect whether the 31st or 63rd bit for the last cornerstone is non-zero.
 *                   (The last cornerstone has a value of nodeRange<I>(0) = 2^31 or 2^63)
 * @return           The of the digit at place @p position
 *
 * The position argument correspondence to octal digit places has been chosen such that
 * octalDigit(code, pos) returns the octant at octree division level pos.
 */
template<class I>
CUDA_HOST_DEVICE_FUN
constexpr unsigned octalDigit(I code, unsigned position)
{
    return (code >> (3u * (maxTreeLevel<I>{} - position))) & 7u;
}

//! @brief cut down the input morton code to the start code of the enclosing box at <treeLevel>
template<class I>
CUDA_HOST_DEVICE_FUN
constexpr std::enable_if_t<std::is_unsigned<I>{}, I> enclosingBoxCode(I code, unsigned treeLevel)
{
    // total usable bits in the morton code, 30 or 63
    constexpr unsigned nBits = 3 * maxTreeLevel<I>{};

    // number of bits to discard, counting from lowest bit
    unsigned discardedBits = nBits - 3 * treeLevel;
    code = code >> discardedBits;
    return code << discardedBits;
}

/*! @brief compute an enclosing envelope corresponding to the smallest possible
 *         octree node for two input SFC codes
 *
 * @tparam I              32- or 64-bit unsigned integer type
 * @param[in] firstCode   lower SFC code
 * @param[in] secondCode  upper SFC code
 *
 * @return                two morton codes that delineate the start and end of
 *                        the smallest octree node that contains both input codes
 */
template<class I>
constexpr pair<I> smallestCommonBox(I firstCode, I secondCode)
{
    assert(firstCode <= secondCode);

    unsigned commonLevel = commonPrefix(firstCode, secondCode) / 3;
    I        nodeStart   = enclosingBoxCode(firstCode, commonLevel);

    return pair<I>(nodeStart, nodeStart + nodeRange<I>(commonLevel));
}

//! @brief zero all but the highest nBits in a SFC code
template<class I>
CUDA_HOST_DEVICE_FUN
constexpr I zeroLowBits(I code, int nBits)
{
    int nLowerBits = sizeof(I) * 8 - unusedBits<I>{} - nBits;
    I mask = (I(1) << nLowerBits) - 1;

    return code & ~mask;
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
    if (x) return maxTreeLevel<I>{} - __builtin_ctzl(x)/3;
    else   return maxTreeLevel<I>{} - 0;
}

/*! @brief return the power of 8 for the octal place at position @p pos
 *
 * @tparam I    32- or 64-bit unsigned integer
 * @param  pos  Position counting from left, starting from 1. Maximum value 10 or 21 (64-bit)
 * @return      the power of 8 associated with the indicated octal place
 */
template<class I>
constexpr I octalPower(int pos)
{
    return (I(1) << 3 * (maxTreeLevel<I>{} - pos));
}

/*! @brief generate SFC codes to cover the range [a:b] with a valid cornerstone sub-octree
 *
 * @tparam     I       32- or 64-bit unsigned integer
 * @param[in]  a       first SFC code
 * @param[in]  b       second SFC code, b > a
 * @param[out] output  output sfc codes
 * @return             number of values in output
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
template<class I, class Store>
std::enable_if_t<std::is_same_v<Store, std::nullptr_t> || std::is_same_v<Store, I*>, int>
spanSfcRange(I a, I b, Store output)
{
    int numValues = 0;
    // position of first differing octal digit place
    int ab_first_diff_pos = (countLeadingZeros(a ^ b) + 3 - unusedBits<I>{}) / 3;

    // last non-zero octal digit place position in a and b
    int a_last_nz_pos = lastNzPlace(a);
    int b_last_nz_pos = lastNzPlace(b);

    // add SFC codes, increasing power of 8 in each iteration
    for (int pos = a_last_nz_pos; pos > ab_first_diff_pos; --pos)
    {
        int numDigits = (8 - octalDigit(a, pos)) % 8;
        numValues += numDigits;
        while (numDigits--)
        {
            if constexpr (!std::is_same_v<Store, std::nullptr_t>) { *output++ = a; }
            a += octalPower<I>(pos);
        }
    }

    // add SFC codes, decreasing power of 8 in each iteration
    for (int pos = ab_first_diff_pos; pos <= b_last_nz_pos; ++pos)
    {
        // Note: octalDigit(a, pos) is guaranteed zero for pos > ab_first_diff_pos
        int numDigits = octalDigit(b, pos) - octalDigit(a, pos);
        numValues += numDigits;
        while (numDigits--)
        {
            if constexpr (!std::is_same_v<Store, std::nullptr_t>) { *output++ = a; }
            a += octalPower<I>(pos);
        }
    }

    return numValues;
}

//! @brief overload to skip storage and just compute number of values, see spanSfcRange(I a, I b, I* output) above
template<class I>
int spanSfcRange(I a, I b)
{
    return spanSfcRange<I, std::nullptr_t>(a, b, nullptr);
}

} // namespace cstone
