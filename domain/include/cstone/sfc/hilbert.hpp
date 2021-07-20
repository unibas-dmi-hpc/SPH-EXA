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
 * This code is based on the implementation of the Hilbert curves presented in:
 *
 * Yohei Miki, Masayuki Umemura
 * GOTHIC: Gravitational oct-tree code accelerated by hierarchical time step controlling
 * https://doi.org/10.1016/j.newast.2016.10.007
 */

#pragma once

#include "morton.hpp"

namespace cstone
{

/*! @brief compute the Hilbert key for a 3D point of integer coordinates
 *
 * @tparam KeyType       32- or 64-bit unsigned integer
 * @param[in]  px,py,pz  input coordinates in [0:2^maxTreeLevel<KeyType>{}]
 * @return               the Hilbert key
 */
template<class KeyType>
CUDA_HOST_DEVICE_FUN inline
std::enable_if_t<std::is_unsigned_v<KeyType>, KeyType> iHilbert(unsigned px, unsigned py, unsigned pz)
{
    assert(px < (1u << maxTreeLevel<KeyType>{}));
    assert(py < (1u << maxTreeLevel<KeyType>{}));
    assert(pz < (1u << maxTreeLevel<KeyType>{}));

    KeyType key = 0;
    for (int level = maxTreeLevel<KeyType>{} - 1; level >= 0; --level)
    {
        unsigned xi = (px >> level) & 1u;
        unsigned yi = (py >> level) & 1u;
        unsigned zi = (pz >> level) & 1u;

        // turn px, py and pz
        px ^= -( xi & ((!yi) |   zi));
        py ^= -((xi & (  yi  |   zi)) | (yi & (!zi)));
        pz ^= -((xi &  (!yi) & (!zi)) | (yi & (!zi)));

        // append 3 bits to the key
        key = (key << 3) + ((xi << 2) | ((xi ^ yi) << 1) | ((xi ^ zi) ^ yi));

        // rotate non-cyclic x->z->y->x
        if (zi)
        {
            unsigned pt = px;
            px = py;
            py = pz;
            pz = pt;
        }
        else if (!yi)
        {
            unsigned pt = px;
            px = pz;
            pz = pt;
        }
    }

    return key;
}

//! @brief inverse function of iHilbert
template<class KeyType>
CUDA_HOST_DEVICE_FUN inline
std::enable_if_t<std::is_unsigned_v<KeyType>> idecodeHilbert(KeyType key, unsigned* rx, unsigned* ry, unsigned* rz)
{
   unsigned px = 0;
   unsigned py = 0;
   unsigned pz = 0;

   for (int level = 0; level < maxTreeLevel<KeyType>{}; ++level)
   {
       unsigned xi = (key >> (3 * level + 2)) & 1u;
       unsigned yi = (key >> (3 * level + 1)) & 1u;
       unsigned zi = (key >> (3 * level    )) & 1u;

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

   *rx = px;
   *ry = py;
   *rz = pz;
}

template<class KeyType>
KeyType iHilbert2(unsigned xx, unsigned yy, unsigned zz)
{
    constexpr unsigned octantMap[8] = {0, 1, 7, 6, 3, 2, 4, 5};
    unsigned mask = 1 << (maxTreeLevel<KeyType>{} - 1);
    KeyType key = 0;

    for (int i = 0; i < maxTreeLevel<KeyType>{}; i++)
    {
        unsigned ix = (xx & mask) ? 1 : 0;
        unsigned iy = (yy & mask) ? 1 : 0;
        unsigned iz = (zz & mask) ? 1 : 0;
        unsigned octant = (ix << 2) + (iy << 1) + iz;
        if (octant == 0)
        {
            std::swap(yy, zz);
        }
        else if (octant == 1 || octant == 5)
        {
            std::swap(xx, yy);
        }
        else if (octant == 4 || octant == 6)
        {
            xx = (xx) ^ 0xFFFFFFFF;
            zz = (zz) ^ 0xFFFFFFFF;
        }
        else if (octant == 3 || octant == 7)
        {
            xx = (xx) ^ 0xFFFFFFFF;
            yy = (yy) ^ 0xFFFFFFFF;
            std::swap(xx, yy);
        }
        else
        {
            yy = (yy) ^ 0xFFFFFFFF;
            zz = (zz) ^ 0xFFFFFFFF;
            std::swap(yy, zz);
        }
        key = (key << 3) + octantMap[octant];
        mask >>= 1;
    }

    return key;
}

} // namespace cstone
