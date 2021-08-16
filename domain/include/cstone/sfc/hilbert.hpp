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
 * @brief  3D Hilbert encoding/decoding in 32- and 64-bit
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * This code is based on the implementation of the Hilbert curve presented in:
 *
 * Yohei Miki, Masayuki Umemura
 * GOTHIC: Gravitational oct-tree code accelerated by hierarchical time step controlling
 * https://doi.org/10.1016/j.newast.2016.10.007
 */

#pragma once

#include "morton.hpp"

namespace cstone
{

#if defined(__CUDACC__) || defined(__HIPCC__)
__device__ static unsigned mortonToHilbertDevice[8] = { 0, 1, 3, 2, 7, 6, 4, 5 };
#endif

/*! @brief compute the Hilbert key for a 3D point of integer coordinates
 *
 * @tparam     KeyType   32- or 64-bit unsigned integer
 * @param[in]  px,py,pz  input coordinates in [0:2^maxTreeLevel<KeyType>{}]
 * @return               the Hilbert key
 */
template<class KeyType>
constexpr HOST_DEVICE_FUN inline
std::enable_if_t<std::is_unsigned_v<KeyType>, KeyType> iHilbert(unsigned px, unsigned py, unsigned pz) noexcept
{
    assert(px < (1u << maxTreeLevel<KeyType>{}));
    assert(py < (1u << maxTreeLevel<KeyType>{}));
    assert(pz < (1u << maxTreeLevel<KeyType>{}));

    #if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
    constexpr unsigned mortonToHilbert[8] = { 0, 1, 3, 2, 7, 6, 4, 5 };
    #endif

    KeyType key = 0;

    for (int level = maxTreeLevel<KeyType>{} - 1; level >= 0; --level)
    {
        unsigned xi = (px >> level) & 1u;
        unsigned yi = (py >> level) & 1u;
        unsigned zi = (pz >> level) & 1u;

        // append 3 bits to the key
        unsigned octant = (xi << 2) | (yi << 1) | zi;
        #if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        key = (key << 3) + mortonToHilbertDevice[octant];
        #else
        key = (key << 3) + mortonToHilbert[octant];
        #endif

        // turn px, py and pz
        px ^= -( xi & ((!yi) |   zi));
        py ^= -((xi & (  yi  |   zi)) | (yi & (!zi)));
        pz ^= -((xi &  (!yi) & (!zi)) | (yi & (!zi)));

        if (zi)
        {
            // cyclic rotation
            unsigned pt = px;
            px = py;
            py = pz;
            pz = pt;
        }
        else if (!yi)
        {
            // swap x and z
            unsigned pt = px;
            px = pz;
            pz = pt;
        }
    }

    return key;
}

//! @brief inverse function of iHilbert
template<class KeyType>
HOST_DEVICE_FUN inline
util::tuple<unsigned, unsigned, unsigned> decodeHilbert(KeyType key) noexcept
{
   unsigned px = 0;
   unsigned py = 0;
   unsigned pz = 0;

   for (int level = 0; level < maxTreeLevel<KeyType>{}; ++level)
   {
       unsigned octant = (key >> (3 * level)) & 7u;
       const unsigned xi = octant >> 2u;
       const unsigned yi = (octant >> 1u) & 1u;
       const unsigned zi = octant & 1u;

       if (yi ^ zi)
       {
       	   // cyclic rotation
           unsigned pt = px;
           px = pz;
           pz = py;
           py = pt;
       }
       else if ((!xi & !yi & !zi) || (xi & yi &zi) )
       {
           // swap x and z
           unsigned pt = px;
           px = pz;
           pz = pt;
       }

       // turn px, py and pz
       unsigned mask = (1 << level) - 1;
       px ^= mask & (-( xi & (  yi   |    zi )));
       py ^= mask & (-((xi & ((!yi)  |  (!zi)))   | ((!xi) & yi & zi)));
       pz ^= mask & (-((xi &  (!yi)  &  (!zi) )   | (        yi & zi)));

       // append 1 bit to the positions
       px |= ( xi       << level);
       py |= ((xi ^ yi) << level);
       pz |= ((yi ^ zi) << level);
   }

   return { px, py, pz };
}

/*! @brief compute the 3D integer coordinate box that contains the key range
 *
 * @tparam KeyType   32- or 64-bit unsigned integer
 * @param  keyStart  lower Hilbert key
 * @param  keyEnd    upper Hilbert key
 * @return           the integer box that contains the given key range
 */
template<class KeyType>
HOST_DEVICE_FUN IBox hilbertIBox(KeyType keyStart, unsigned level) noexcept
{
    assert(level <= maxTreeLevel<KeyType>{});
    constexpr unsigned maxCoord = 1u << maxTreeLevel<KeyType>{};
    unsigned cubeLength = maxCoord >> level;
    unsigned mask = ~(cubeLength - 1);

    auto [ix, iy, iz] = decodeHilbert(keyStart);

    // round integer coordinates down to corner closest to origin
    ix &= mask;
    iy &= mask;
    iz &= mask;

    return IBox(ix, ix + cubeLength, iy, iy + cubeLength, iz, iz + cubeLength);
}

//! @brief convenience wrapper
template<class KeyType>
HOST_DEVICE_FUN IBox hilbertIBoxKeys(KeyType keyStart, KeyType keyEnd) noexcept
{
    assert(keyStart <= keyEnd);
    return hilbertIBox(keyStart, treeLevel(keyEnd - keyStart));
}

} // namespace cstone
