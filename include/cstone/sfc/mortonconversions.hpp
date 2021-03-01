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
 * \brief utility functions for Morton code conversions
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * These conversion functions can be used to generate Morton
 * codes from octree indices and from integer x,y,z coordinates.
 */

#pragma once

#include <array>

#include "mortoncode.hpp"

namespace cstone
{

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

/*! \brief Decode a morton code into x,y,z and convert coordinates
 *
 * \tparam I         32- or 64-bit unsigned integer
 * \param code       input Morton code
 * \param treeLevel  octree subdivision level
 * \return           array with x,y,z in the range [0, 2^treeLevel-1]
 */
template<class I>
std::array<unsigned, 3> boxFromCode(I code, unsigned treeLevel)
{
    // 10 or 21 bits per dimension
    constexpr unsigned nBits = (sizeof(I) * 8) / 3;

    // number of bits to discard
    unsigned discardedBits = nBits - treeLevel;

    code = enclosingBoxCode(code, treeLevel);

    I x = decodeMortonX(code);
    I y = decodeMortonY(code);
    I z = decodeMortonZ(code);

    return {unsigned(x >> discardedBits), unsigned(y >> discardedBits), unsigned(z >> discardedBits)};
}

/*! \brief transfer a series of hierarchical octree indices into a morton code
 *
 * \tparam I       32- or 64-bit unsigned integer
 * \param indices  indices[0] contains the octree index 0-7 for the top-level,
 *                 indices[1] refers to the first subdivision, etc
 *                 a 32-bit integer can resolve up to 10 layers, while
 *                 a 64-bit integer can resolve 21 layers
 *
 *                 Note: all indices must be in the range [0-7]!
 *
 * \return         the morton code
 */
template<class I>
inline I codeFromIndices(std::array<unsigned char, maxTreeLevel<uint64_t>{}> indices)
{
    constexpr unsigned nLevels = (sizeof(I) * 8) / 3;

    I ret = 0;
    for(unsigned idx = 0; idx < nLevels; ++idx)
    {
        assert(indices[idx] < 8);
        unsigned treeLevel = nLevels - idx - 1;
        ret += I(indices[idx]) << (3*treeLevel);
    }

    return ret;
}

/*! \brief convert a morton code into a series of hierarchical octree indices
 *
 * \tparam I       32- or 64-bit unsigned integer
 * \param[in]      the morton code
 *
 * \return indices indices[0] contains the octree index 0-7 for the top-level,
 *                 indices[1] refers to the first subdivision, etc
 *                 a 32-bit integer can resolve up to 10 layers, while
 *                 a 64-bit integer can resolve 21 layers
 */
template<class I>
inline std::array<unsigned char, maxTreeLevel<I>{}> indicesFromCode(I code)
{
    constexpr unsigned nLevels = maxTreeLevel<I>{};

    std::array<unsigned char, nLevels> ret;
    for(unsigned idx = 0; idx < nLevels; ++idx)
    {
        unsigned treeLevel = nLevels - idx - 1;
        ret[treeLevel] = (code >> (3*idx)) % 8;
    }

    return ret;
}

} // namespace cstone
