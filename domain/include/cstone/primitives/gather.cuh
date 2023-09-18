/*
 * MIT License
 *
 * Copyright (c) 2022 CSCS, ETH Zurich
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
 * @brief  Exposes gather functionality to reorder arrays with a map
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <cassert>
#include <cstdint>
#include <memory>

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/primitives/primitives_gpu.h"
#include "cstone/util/reallocate.hpp"

namespace cstone
{

template<class IndexType, class BufferType>
class GpuSfcSorter
{
public:
    GpuSfcSorter(BufferType& buffer)
        : buffer_(buffer)
    {
    }

    GpuSfcSorter(const GpuSfcSorter&) = delete;

    const IndexType* getMap() const { return ordering(); }

    /*! @brief sort given Morton codes on the device and determine reorder map based on sort order
     *
     * @param[inout] first   pointer to first SFC code
     * @param[inout] last    pointer to last SFC code
     *
     * Precondition:
     *   - [first:last] is a continuous sequence of accessible elements
     *
     * Postcondition
     *   - [first:last] is sorted
     *   - subsequent calls to operator() apply a gather operation to the input sequence
     *     with the map obtained from sort_by_key with [first:last] as the keys
     *     and the identity permutation as the values
     */
    template<class KeyType>
    void setMapFromCodes(KeyType* first, KeyType* last)
    {
        mapSize_ = std::size_t(last - first);
        reallocateBytes(buffer_, mapSize_ * sizeof(IndexType));
        sequenceGpu(ordering(), mapSize_, LocalIndex(0));
        sortByKeyGpu(first, last, ordering());
    }

    template<class KeyType>
    void setMapFromCodes(KeyType* first, KeyType* last, KeyType* keyBuf, IndexType* valueBuf)
    {
        mapSize_ = std::size_t(last - first);
        reallocateBytes(buffer_, mapSize_ * sizeof(IndexType));
        sequenceGpu(ordering(), mapSize_, LocalIndex(0));
        sortByKeyGpu(first, last, ordering(), keyBuf, valueBuf);
    }

    template<class KeyType, class KeyBuf, class ValueBuf>
    void setMapFromCodes(KeyType* first, KeyType* last, KeyBuf& keyBuf, ValueBuf& valueBuf)
    {
        mapSize_ = std::size_t(last - first);
        reallocateBytes(buffer_, mapSize_ * sizeof(IndexType));
        sequenceGpu(ordering(), mapSize_, LocalIndex(0));

        auto s1 = reallocateBytes(keyBuf, mapSize_ * sizeof(KeyType));
        auto s2 = reallocateBytes(valueBuf, mapSize_ * sizeof(IndexType));

        setMapFromCodes(first, last, (KeyType*)rawPtr(keyBuf), (IndexType*)rawPtr(valueBuf));

        reallocateDevice(keyBuf, s1, 1.01);
        reallocateDevice(valueBuf, s2, 1.01);
    }

    auto gatherFunc() const { return gatherGpuL; }

private:
    IndexType* ordering() { return reinterpret_cast<IndexType*>(rawPtr(buffer_)); }
    const IndexType* ordering() const { return reinterpret_cast<const IndexType*>(rawPtr(buffer_)); }

    //! @brief reference to (non-owning) buffer for ordering
    BufferType& buffer_;
    std::size_t mapSize_{0};
};

} // namespace cstone
