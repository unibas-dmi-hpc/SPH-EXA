#pragma once

#include <array>

#include "sfc/mortoncode.hpp"

/*! \brief \file utility functions for Morton code conversions
 *
 * These conversion functions can be used to generate Morton
 * codes from octree indices and from integer x,y,z coordinates.
 */

namespace sphexa
{

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
inline I codeFromIndices(std::array<unsigned char, 21> indices)
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

} // namespace sphexa
