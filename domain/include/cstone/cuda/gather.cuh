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

template<class LocalIndex>
class DeviceMemory;

/*! @brief A stateful functor for reordering arrays on the gpu
 *
 * @tparam ValueType   any type that behaves like a built-in type (check template instantation list)
 * @tparam KeyType     32- or 64-bit unsigned integer
 * @tparam IndexType   type to index node-local particles, 32-bit or 64-bit integer
 */
template<class KeyType, class IndexType>
class DeviceSfcSort
{
public:

    DeviceSfcSort();

    ~DeviceSfcSort();

    //! @brief download the reorder map from the device
    const IndexType* getReorderMap() const;

    /*! @brief sort given Morton codes on the device and determine reorder map based on sort order
     *
     * \param[inout] firstKey   pointer to first Morton code
     * \param[inout] lastKey    pointer to last Morton code
     *
     * Precondition:
     *   - [codes_first:codes_last] is a continues sequence of accessible elements of size N
     *
     * Postcondition
     *   - [codes_first:codes_last] is sorted
     *   - subsequent calls to operator() apply a gather operation to the input sequence
     *     with the map obtained from sort_by_key with [codes_first:codes_last] as the keys
     *     and the identity permutation as the values
     *
     *  Remarks:
     *    - reallocates space on the device if necessary to fit N elements of type IndexType
     *      and a second buffer of size max(2N*sizeof(T), N*sizeof(KeyType))
     */
    void setMapFromCodes(KeyType* firstKey, KeyType* lastKey);

    /*! @brief reorder the array \a values according to the reorder map provided previously
     *
     * \a values must have at least as many elements as the reorder map provided in the last call
     * to setReorderMap or setMapFromCodes, otherwise the behavior is undefined.
     */
    template<class T>
    void operator()(const T* values, T* destination, IndexType offset, IndexType numExtract) const;

    template<class T>
    void operator()(const T* values, T* destination) const;

    void restrictRange(std::size_t offset, std::size_t numExtract);

private:
    std::size_t offset_{0};
    std::size_t numExtract_{0};
    std::size_t mapSize_{0};

    std::unique_ptr<DeviceMemory<IndexType>> deviceMemory_;
};

extern template class DeviceSfcSort<unsigned, unsigned>;
extern template class DeviceSfcSort<uint64_t, unsigned>;

template<class IndexType, class BufferType>
class DeviceSfcSortRef
{
public:
    DeviceSfcSortRef(BufferType& buffer)
        : buffer_(buffer)
    {
    }

    DeviceSfcSortRef(const DeviceSfcSortRef&) = delete;

    const IndexType* getReorderMap() const { return ordering(); }

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
        offset_     = 0;
        mapSize_    = std::size_t(last - first);
        numExtract_ = mapSize_;

        reallocateBytes(buffer_, mapSize_ * sizeof(IndexType));
        sequenceGpu(ordering(), mapSize_, 0u);
        sortByKeyGpu(first, last, ordering());
    }

    /*! @brief reorder the array @a values according to the reorder map provided previously
     *
     * @a values must have at least as many elements as the reorder map provided in the last call
     * to setReorderMap or setMapFromCodes, otherwise the behavior is undefined.
     */
    template<class T>
    void operator()(const T* source, T* destination, IndexType offset, IndexType numExtract) const
    {
        gatherGpu(ordering() + offset, numExtract, source, destination);
    }

    template<class T>
    void operator()(const T* source, T* destination) const
    {
        this->operator()(source, destination, offset_, numExtract_);
    }

    void restrictRange(std::size_t offset, std::size_t numExtract)
    {
        assert(offset + numExtract <= mapSize_);

        offset_     = offset;
        numExtract_ = numExtract;
    }

private:
    IndexType* ordering() { return reinterpret_cast<IndexType*>(rawPtr(buffer_)); }
    const IndexType* ordering() const { return reinterpret_cast<const IndexType*>(rawPtr(buffer_)); }

    std::size_t offset_{0};
    std::size_t numExtract_{0};
    std::size_t mapSize_{0};

    //! @brief reference to (non-owning) buffer for ordering
    BufferType& buffer_;
};

} // namespace cstone
