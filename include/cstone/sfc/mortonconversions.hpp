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
