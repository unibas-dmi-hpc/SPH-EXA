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
 * @brief  SFC encoding/decoding in 32- and 64-bit
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * Common interface to Morton and Hilbert keys based on strong C++ types
 */

#pragma once

#include "morton.hpp"
#include "hilbert.hpp"

namespace cstone
{

//! @brief Strong type for Morton keys
template<class IntegerType>
using MortonKey = StrongType<IntegerType, struct MortonKeyTag>;

//! @brief Strong type for Hilbert keys
template<class IntegerType>
using HilbertKey = StrongType<IntegerType, struct HilbertKeyTag>;

//! @brief Meta function to detect Morton key types
template<class KeyType>
struct IsMorton : std::bool_constant<std::is_same_v<KeyType, MortonKey<typename KeyType::ValueType>>> {};

//! @brief Meta function to detect Hilbert key types
template<class KeyType>
struct IsHilbert : std::bool_constant<std::is_same_v<KeyType, HilbertKey<typename KeyType::ValueType>>> {};

//! @brief Key encode overload for Morton keys
template<class KeyType>
HOST_DEVICE_FUN inline std::enable_if_t<IsMorton<KeyType>{}, KeyType>
iSfcKey(unsigned ix, unsigned iy, unsigned iz)
{
    return KeyType{iMorton<typename KeyType::ValueType>(ix, iy, iz)};
}

//! @brief Key encode overload for Hilbert keys
template<class KeyType>
HOST_DEVICE_FUN inline std::enable_if_t<IsHilbert<KeyType>{}, KeyType>
iSfcKey(unsigned ix, unsigned iy, unsigned iz)
{
    return KeyType{iHilbert<typename KeyType::ValueType>(ix, iy, iz)};
}

template <class KeyType, class T>
HOST_DEVICE_FUN inline KeyType sfc3D(T x, T y, T z, T xmin, T ymin, T zmin, T mx, T my, T mz)
{
    constexpr unsigned mcoord = (1u << maxTreeLevel<typename KeyType::ValueType>{}) - 1;

    unsigned ix = stl::min(unsigned((x - xmin) * mx), mcoord);
    unsigned iy = stl::min(unsigned((y - ymin) * my), mcoord);
    unsigned iz = stl::min(unsigned((z - zmin) * mz), mcoord);

    return iSfcKey<KeyType>(ix, iy, iz);
}

/*! @brief Calculates a Hilbert key for a 3D point within the specified box
 *
 * @tparam    KeyType specify either a 32 or 64 bit unsigned integer Morton or Hilbert key type.
 * @param[in] x,y,z   input coordinates within the unit cube [0,1]^3
 * @param[in] box     bounding for coordinates
 * @return            the SFC key
 *
 * Note: -KeyType needs to be specified explicitly.
 *       -not specifying an unsigned type results in a compilation error
 */
template <class KeyType, class T>
HOST_DEVICE_FUN inline KeyType sfc3D(T x, T y, T z, const Box<T>& box)
{
    constexpr unsigned cubeLength = (1u << maxTreeLevel<typename KeyType::ValueType>{});

    return sfc3D<KeyType>(x, y, z, box.xmin(), box.ymin(), box.zmin(),
                          cubeLength * box.ilx(), cubeLength * box.ily(), cubeLength * box.ilz());
}

//! @brief decode a Morton key
template<class KeyType>
HOST_DEVICE_FUN inline std::enable_if_t<IsMorton<KeyType>{}, util::tuple<unsigned, unsigned, unsigned>>
decodeSfc(KeyType key)
{
    return decodeMorton<typename KeyType::ValueType>(key);
}

//! @brief decode a Hilbert key
template<class KeyType>
HOST_DEVICE_FUN inline std::enable_if_t<IsHilbert<KeyType>{}, util::tuple<unsigned, unsigned, unsigned>>
decodeSfc(KeyType key)
{
    return decodeHilbert<typename KeyType::ValueType>(key);
}

/*! @brief compute the Morton codes for the input coordinate arrays
 *
 * @tparam     T          float or double
 * @param[in]  xBegin     input iterators for coordinate arrays
 * @param[in]  xEnd
 * @param[in]  yBegin
 * @param[in]  zBegin
 * @param[out] codeBegin  output for morton codes
 * @param[in]  box        coordinate bounding box
 */
template<class InputIterator, class OutputIterator, class T>
void computeMortonCodes(InputIterator  xBegin,
                        InputIterator  xEnd,
                        InputIterator  yBegin,
                        InputIterator  zBegin,
                        OutputIterator codesBegin,
                        const Box<T>& box)
{
    assert(xEnd >= xBegin);
    using KeyType = std::decay_t<decltype(*codesBegin)>;

    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < std::size_t(xEnd-xBegin); ++i)
    {
        codesBegin[i] = sfc3D<MortonKey<KeyType>>(xBegin[i], yBegin[i], zBegin[i], box);
    }
}

} // namespace cstone
