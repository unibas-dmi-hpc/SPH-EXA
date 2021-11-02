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
 * @brief Functionality for calculating for performing gather operations on the CPU
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */


#pragma once

#include <algorithm>
#include <cassert>
#include <numeric>
#include <tuple>
#include <vector>

#include "cstone/sfc/morton.hpp"
#include "cstone/util/gsl-lite.hpp"

namespace cstone
{

/*! @brief sort values according to a key
 *
 * @param[inout] keyBegin    key sequence start
 * @param[inout] keyEnd      key sequence end
 * @param[inout] valueBegin  values
 * @param[in]    compare     comparison function
 *
 * Upon completion of this routine, the key sequence will be sorted and values
 * will be rearranged to reflect the key ordering
 */
template <class InoutIterator, class OutputIterator, class Compare>
void sort_by_key(InoutIterator keyBegin, InoutIterator keyEnd, OutputIterator valueBegin, Compare compare)
{
    using KeyType   = std::decay_t<decltype(*keyBegin)>;
    using ValueType = std::decay_t<decltype(*valueBegin)>;
    std::size_t n   = std::distance(keyBegin, keyEnd);

    // zip the input integer array together with the index sequence
    std::vector<std::tuple<KeyType, ValueType>> keyIndexPairs(n);
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i)
        keyIndexPairs[i] = std::make_tuple(keyBegin[i], valueBegin[i]);

    // sort, comparing only the first tuple element
    std::sort(begin(keyIndexPairs), end(keyIndexPairs),
              [compare](const auto& t1, const auto& t2){ return compare(std::get<0>(t1), std::get<0>(t2)); });

    // extract the resulting ordering and store back the sorted keys
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i)
    {
        keyBegin[i]  = std::get<0>(keyIndexPairs[i]);
        valueBegin[i] = std::get<1>(keyIndexPairs[i]);
    }
}

//! @brief calculate the sortKey that sorts the input sequence, default ascending order
template <class InoutIterator, class OutputIterator>
void sort_by_key(InoutIterator inBegin, InoutIterator inEnd, OutputIterator outBegin)
{
    sort_by_key(inBegin, inEnd, outBegin, std::less<std::decay_t<decltype(*inBegin)>>{});
}

template<class IndexType, class ValueType>
void reorder(gsl::span<const IndexType> ordering, const ValueType* source, ValueType* destination,
             IndexType offset, IndexType numExtract)
{
    #pragma omp parallel for schedule(static)
    for (IndexType i = 0; i < numExtract; ++i)
    {
        destination[i] = source[ordering[i + offset]];
    }
}


/*! @brief reorder the input array according to the specified ordering, no reallocation
 *
 * @tparam LocalIndex    integer type
 * @tparam ValueType     float or double
 * @param ordering       an ordering
 * @param array          an array, size >= ordering.size()
 */
template<class LocalIndex, class ValueType>
void reorderInPlace(const std::vector<LocalIndex>& ordering, ValueType* array)
{
    std::vector<ValueType> tmp(ordering.size());
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < ordering.size(); ++i)
    {
        tmp[i] = array[ordering[i]];
    }
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < ordering.size(); ++i)
    {
        array[i] = tmp[i];
    }
}

//! @brief This class conforms to the same interface as the device version to allow abstraction
template<class ValueType, class CodeType, class IndexType>
class CpuGather
{
public:
    CpuGather() = default;

    /*! @brief upload the new reorder map to the device and reallocates buffers if necessary
     *
     * If the sequence [map_first:map_last] does not contain each element [0:map_last-map_first]
     * exactly once, the behavior is undefined.
     */
    void setReorderMap(const IndexType* map_first, const IndexType* map_last)
    {
        mapSize_ = std::size_t(map_last - map_first);
        ordering_.resize(mapSize_);
        std::copy(map_first, map_last, begin(ordering_));

        buffer_.resize(mapSize_);
    }

    void getReorderMap(IndexType* map_first)
    {
        std::copy(ordering_.data(), ordering_.data() + mapSize_, map_first);
    }

    /*! @brief sort given Morton codes on the device and determine reorder map based on sort order
     *
     * @param[inout] codes_first   pointer to first Morton code
     * @param[inout] codes_last    pointer to last Morton code
     *
     * Precondition:
     *   - [codes_first:codes_last] is a continues sequence of accessible elements of size N
     *
     * Postcondition:
     *   - [codes_first:codes_last] is sorted
     *   - subsequent calls to operator() apply a gather operation to the input sequence
     *     with the map obtained from sort_by_key with [codes_first:codes_last] as the keys
     *     and the identity permutation as the values
     *
     *  Remarks:
     *    - reallocates space on the device if necessary to fit N elements of type LocalParticleIndex
     *      and a second buffer of size max(2N*sizeof(T), N*sizeof(KeyType))
     */
    void setMapFromCodes(CodeType* codes_first, CodeType* codes_last)
    {
        mapSize_ = std::size_t(codes_last - codes_first);
        ordering_.resize(mapSize_);
        std::iota(begin(ordering_), end(ordering_), 0);

        sort_by_key(codes_first, codes_last, begin(ordering_));

        buffer_.resize(mapSize_);
    }

    /*! @brief reorder the array @p values according to the reorder map provided previously
     *
     * @p values must have at least as many elements as the reorder map provided in the last call
     * to setReorderMap or setMapFromCodes, otherwise the behavior is undefined.
     */
    void operator()(const ValueType* source, ValueType* destination, IndexType offset, IndexType numExtract)
    {
        reorder<IndexType>(ordering_, source, buffer_.data(), 0, mapSize_);

        #pragma omp parallel for schedule(static)
        for (IndexType i = 0; i < numExtract; ++i)
        {
            destination[i] = buffer_[i + offset];
        }
        //reorderInPlace(ordering_, values);
    }

private:
    std::size_t mapSize_{0};
    std::vector<IndexType> ordering_;

    std::vector<ValueType> buffer_;
};

} // namespace cstone
