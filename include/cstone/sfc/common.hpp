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
 * \brief  Common operations on SFC keys that do not depend on the specific SFC used
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <cstdint>
#include <type_traits>

#include "cstone/primitives/clz.hpp"
#include "cstone/sfc/box.hpp"
#include "cstone/util.hpp"

namespace cstone
{
//! \brief number of unused leading zeros in a 32-bit Morton code
template<class I>
struct unusedBits : stl::integral_constant<unsigned, 2> {};

//! \brief number of unused leading zeros in a 64-bit Morton code
template<>
struct unusedBits<uint64_t> : stl::integral_constant<unsigned, 1> {};

template<class I>
struct maxTreeLevel : stl::integral_constant<unsigned, 10> {};

template<>
struct maxTreeLevel<uint64_t> : stl::integral_constant<unsigned, 21> {};

namespace detail
{

/*! \brief normalize a floating point number in [0,1] to an integer in [0, 2^(10 or 21)-1]
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
inline I toNBitInt(T x)
{
    // spatial resolution in bits per dimension
    constexpr unsigned nBits = maxTreeLevel<I>{};

    // [0,1] to [0,1023] and convert to integer (32-bit) or
    // [0,1] to [0,2097151] and convert to integer (64-bit)
    //return std::min(std::max(x * T(1u<<nBits), T(0.0)), T((1u<<nBits)-1u));
    return stl::min(stl::max(x * T(1u<<nBits), T(0.0)), T((1u<<nBits)-1u));
}

/*! \brief normalize a floating point number in [0,1] to an integer in [0, 2^(10 or 21)-1]
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
inline I toNBitIntCeil(T x)
{
    // spatial resolution in bits per dimension
    constexpr unsigned nBits = maxTreeLevel<I>{};

    // [0,1] to [0,1023] and convert to integer (32-bit) or
    // [0,1] to [0,2097151] and convert to integer (64-bit)
    return stl::min(stl::max(std::ceil(x * T(1u<<nBits)), T(0.0)), T((1u<<nBits)-1u));
}

}

/*! \brief add (binary) zeros behind a prefix
 *
 * Allows comparisons, such as number of leading common bits (cpr)
 * of the prefix with Morton codes.
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
 *  i.e. \a length plus the number of zeros added adds up to 30 for 32-bit integers
 *  or 63 for 64-bit integers, because these are the numbers of usable bits in Morton codes.
 */
template <class I>
constexpr I pad(I prefix, int length)
{
    return prefix << (3*maxTreeLevel<I>{} - length);
}

/*! \brief compute the maximum range of an octree node at a given subdivision level
 *
 * \tparam I         32- or 64-bit unsigned integer type
 * \param treeLevel  octree subdivision level
 * \return           the range
 *
 * At treeLevel 0, the range is the entire 30 or 63 bits used in the Morton code.
 * After that, the range decreases by 3 bits for each level.
 *
 */
template<class I>
CUDA_HOST_DEVICE_FUN
inline std::enable_if_t<std::is_unsigned<I>{}, I>
nodeRange(unsigned treeLevel)
{
    assert (treeLevel <= maxTreeLevel<I>{});
    unsigned shifts = maxTreeLevel<I>{} - treeLevel;

    I ret = I(1) << (3u * shifts);
    return ret;
}

//! \brief compute ceil(log8(n))
template<class I>
CUDA_HOST_DEVICE_FUN
inline std::enable_if_t<std::is_unsigned<I>{}, unsigned> log8ceil(I n)
{
    if (n == 0)
        return 0;

    unsigned lz = countLeadingZeros(n-1);
    return maxTreeLevel<I>{} - (lz - unusedBits<I>{}) / 3;
}

//! \brief check whether n is a power of 8
template<class I>
CUDA_HOST_DEVICE_FUN
inline std::enable_if_t<std::is_unsigned<I>{}, bool> isPowerOf8(I n)
{
    unsigned lz = countLeadingZeros(n - 1) - unusedBits<I>{};
    return lz % 3 == 0 && !(n & (n-1));
}

/*! \brief calculate common prefix (cpr) of two morton keys
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
int commonPrefix(I key1, I key2)
{
    return int(countLeadingZeros(key1 ^ key2)) - unusedBits<I>{};
}

/*! \brief return octree subdivision level corresponding to codeRange
 *
 * \tparam I         32- or 64-bit unsigned integer type
 * \param codeRange  input Morton code range
 * \return           octree subdivision level 0-10 (32-bit) or 0-21 (64-bit)
 */
template<class I>
CUDA_HOST_DEVICE_FUN
inline unsigned treeLevel(I codeRange)
{
    assert( isPowerOf8(codeRange) );
    return (countLeadingZeros(codeRange - 1) - unusedBits<I>{}) / 3;
}

/*! \brief return the node index between 0-7 of the input code in the parent node
 *
 * \tparam I    32- or 64-bit unsigned integer type
 * \param code  input code corresponding to an octree node
 * \param level octree subdivision level to fully specify the octree node together with @a code
 * \return      the index between 0 and 7 that locates @a code in its enclosing parent node
 *              at level - 1. For the root node at level 0 which has no parent, the return value
 *              is 0.
 */
template<class I>
CUDA_HOST_DEVICE_FUN
inline unsigned parentIndex(I code, unsigned level)
{
    return (code >> (3u * (maxTreeLevel<I>{} - level))) & 7u;
}

//! \brief cut down the input morton code to the start code of the enclosing box at <treeLevel>
template<class I>
CUDA_HOST_DEVICE_FUN
inline std::enable_if_t<std::is_unsigned<I>{}, I> enclosingBoxCode(I code, unsigned treeLevel)
{
    // total usable bits in the morton code, 30 or 63
    constexpr unsigned nBits = 3 * ((sizeof(I) * 8) / 3);

    // number of bits to discard, counting from lowest bit
    unsigned discardedBits = nBits - 3 * treeLevel;
    code = code >> discardedBits;
    return code << discardedBits;
}

/*! \brief compute an enclosing envelope corresponding to the smallest possible
 *         octree node for two input Morton codes
 *
 * \tparam I              32- or 64-bit unsigned integer type
 * \param[in] firstCode   lower Morton code
 * \param[in] secondCode  upper Morton code
 *
 * \return                two morton codes that delineate the start and end of
 *                        the smallest octree node that contains both input codes
 */
template<class I>
inline pair<I> smallestCommonBox(I firstCode, I secondCode)
{
    assert(firstCode <= secondCode);

    unsigned commonLevel = commonPrefix(firstCode, secondCode) / 3;
    I        nodeStart   = enclosingBoxCode(firstCode, commonLevel);

    return pair<I>(nodeStart, nodeStart + nodeRange<I>(commonLevel));
}

//! \brief zero all but the highest nBits in a Morton code
template<class I>
CUDA_HOST_DEVICE_FUN
inline I zeroLowBits(I code, int nBits)
{
    int nLowerBits = sizeof(I) * 8 - unusedBits<I>{} - nBits;
    I mask = (I(1) << nLowerBits) - 1;

    return code & ~mask;
}

} // namespace cstone