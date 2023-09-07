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
 * @brief Buffer description and management for domain decomposition particle exchanges
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/domain/index_ranges.hpp"
#include "cstone/primitives/gather.hpp"
#include "cstone/primitives/math.hpp"
#include "cstone/tree/definitions.h"
#include "cstone/util/tuple_util.hpp"
#include "cstone/util/type_list.hpp"

namespace cstone
{

/*! @brief Layout description for particle buffers
 *
 * Common usage: storing the sub-range of locally owned/assigned particles within particle buffers
 *
 * 0       start              end      size
 * |-------|------------------|--------|
 *   halos   locally assigned   halos
 */
struct BufferDescription
{
    //! @brief subrange start
    LocalIndex start;
    //! @brief subrange end
    LocalIndex end;
    //! @brief total size of the buffer
    LocalIndex size;
};

template<class... Arrays>
auto computeByteOffsets(size_t count, int alignment, Arrays... arrays)
{
    static_assert((... && std::is_pointer_v<Arrays>)&&"all arrays must be pointers");

    util::array<size_t, sizeof...(Arrays) + 1> byteOffsets{sizeof(std::decay_t<decltype(*arrays)>)...};
    byteOffsets *= count;

    //! each sub-buffer will be aligned on a @a alignment-byte aligned boundary
    for (size_t i = 0; i < byteOffsets.size(); ++i)
    {
        byteOffsets[i] = round_up(byteOffsets[i], alignment);
    }

    std::exclusive_scan(byteOffsets.begin(), byteOffsets.end(), byteOffsets.begin(), size_t(0));

    return byteOffsets;
}

template<int alignment, class... Arrays>
size_t computeTotalSendBytes(const SendList& sendList, int thisRank, size_t numBytesHeader, Arrays... arrays)
{
    size_t totalSendBytes = 0;
    for (int destinationRank = 0; destinationRank < int(sendList.size()); ++destinationRank)
    {
        size_t sendCount = sendList[destinationRank].totalCount();
        if (destinationRank == thisRank || sendCount == 0) { continue; }

        size_t particleBytes = computeByteOffsets(sendCount, alignment, arrays...).back();
        //! we add @a alignment bytes to the start of each message to provide space for the message length
        totalSendBytes += numBytesHeader + particleBytes;
    }

    return totalSendBytes;
}

template<int alignment, class... Arrays>
size_t computeTotalSendBytes(const SendRanges& sends, int thisRank, size_t numBytesHeader, Arrays... arrays)
{
    size_t totalSendBytes = 0;
    for (int destinationRank = 0; destinationRank < sends.numRanks(); ++destinationRank)
    {
        size_t sendCount = sends.count(destinationRank);
        if (destinationRank == thisRank || sendCount == 0) { continue; }

        size_t particleBytes = computeByteOffsets(sendCount, alignment, arrays...).back();
        //! we add @a alignment bytes to the start of each message to provide space for the message length
        totalSendBytes += numBytesHeader + particleBytes;
    }

    return totalSendBytes;
}

namespace aux_traits
{
template<class T>
using MakeL2ArrayPtrs = util::array<T*, 2>;
}

/*! @brief Compute multiple pointers such that the argument @p arrays can be mapped into a single buffer
 *
 * @tparam alignment        byte alignment of the individual arrays in the packed buffer
 * @tparam Arrays           arbitrary type of size 16 bytes or smaller
 * @param packedBufferBase  base address of the packed buffer
 * @param arraySize         number of elements of each @p array
 * @param arrays            independent array pointers
 * @return                  a tuple with (src, packed) pointers for each array
 *
 * @p arrays:    ------      ------         ------
 *   (pointers)  a           b              c
 *
 * packedBuffer    |------|------|------|
 *  (pointer)       A      B      C
 *                  |
 *                  packedBufferBase
 *
 * return  tuple( (a, A), (b, B), (c, C) )
 *
 * Pointer types are util::array<float, sizeof(*(a, b or c) / sizeof(float)>..., i.e. the same size as the original
 * element types. This is done to express all types up to 16 bytes with just four util::array types in order
 * to reduce the number of gather/scatter GPU kernel template instantiations.
 */
template<int alignment, class... Arrays>
auto packBufferPtrs(char* packedBufferBase, size_t arraySize, Arrays... arrays)
{
    static_assert((... && std::is_pointer_v<Arrays>)&&"all arrays must be pointers");
    constexpr int numArrays = sizeof...(Arrays);
    constexpr util::array<size_t, numArrays> elementSizes{sizeof(std::decay_t<decltype(*arrays)>)...};
    constexpr auto indices = util::makeIntegralTuple(std::make_index_sequence<numArrays>{});

    const std::array<char*, numArrays> data{reinterpret_cast<char*>(arrays)...};

    auto arrayByteOffsets = computeByteOffsets(arraySize, alignment, arrays...);

    using Types     = util::TypeList<util::array<float, sizeof(std::decay_t<decltype(*arrays)>) / sizeof(float)>...>;
    using PtrTypes  = util::Map<aux_traits::MakeL2ArrayPtrs, Types>;
    using TupleType = util::Reduce<std::tuple, PtrTypes>;

    TupleType ret;

    auto packOneBuffer = [packedBufferBase, &data, &elementSizes, &arrayByteOffsets, &ret](auto index)
    {
        using ElementType = util::array<float, elementSizes[index] / sizeof(float)>;
        auto* srcPtr      = reinterpret_cast<ElementType*>(data[index]);
        auto* packedPtr   = reinterpret_cast<ElementType*>(packedBufferBase + arrayByteOffsets[index]);

        std::get<index>(ret) = util::array<ElementType*, 2>{srcPtr, packedPtr};
    };

    util::for_each_tuple(packOneBuffer, indices);

    return ret;
}

//! @brief Gather @p numElements of each array accessed through @p ordering into @p buffer. CPU and GPU.
template<int alignment, class F, class... Arrays>
std::size_t packArrays(F&& gather, const LocalIndex* ordering, LocalIndex numElements, char* buffer, Arrays... arrays)
{
    auto gatherArray = [&gather, numElements, ordering](auto arrayPair)
    { gather(ordering, numElements, arrayPair[0], arrayPair[1]); };

    auto packTuple = packBufferPtrs<alignment>(buffer, numElements, arrays...);
    util::for_each_tuple(gatherArray, packTuple);

    std::size_t numBytesPacked = computeByteOffsets(numElements, alignment, arrays...).back();
    return numBytesPacked;
}

namespace domain_exchange
{

//! @brief return the required buffer size for calling exchangeParticles
[[maybe_unused]] static LocalIndex
exchangeBufferSize(BufferDescription bufDesc, LocalIndex numPresent, LocalIndex numAssigned)
{
    LocalIndex numIncoming = numAssigned - numPresent;

    bool fitHead = bufDesc.start >= numIncoming;
    bool fitTail = bufDesc.size - bufDesc.end >= numIncoming;

    return (fitHead || fitTail) ? bufDesc.size : bufDesc.end + numIncoming;
}

//! @brief return the index where particles from remote ranks will be received
[[maybe_unused]] static LocalIndex
receiveStart(BufferDescription bufDesc, LocalIndex numPresent, LocalIndex numAssigned)
{
    LocalIndex numIncoming = numAssigned - numPresent;

    bool fitHead = bufDesc.start >= numIncoming;
    assert(fitHead || /*fitTail*/ bufDesc.size - bufDesc.end >= numIncoming);

    if (fitHead) { return bufDesc.start - numIncoming; }
    else { return bufDesc.end; }
}

//! @brief The index range that contains the locally assigned particles. Can contain left-over particles too.
[[maybe_unused]] static util::array<LocalIndex, 2>
assignedEnvelope(BufferDescription bufDesc, LocalIndex numPresent, LocalIndex numAssigned)
{
    LocalIndex numIncoming = numAssigned - numPresent;

    bool fitHead = bufDesc.start >= numIncoming;
    assert(fitHead || /*fitTail*/ bufDesc.size - bufDesc.end >= numIncoming);

    if (fitHead) { return {bufDesc.start - numIncoming, bufDesc.end}; }
    else { return {bufDesc.start, bufDesc.end + numIncoming}; }
}

template<class It>
LocalIndex findInLog(It first, It last, int rank)
{
    auto it = std::find_if(first, last, [rank](auto e) { return std::get<0>(e) == rank; });
    assert(it != last);
    return std::get<1>(*it);
}

template<class Vector>
void extractLocallyOwnedImpl(BufferDescription bufDesc,
                             LocalIndex numPresent,
                             LocalIndex numAssigned,
                             const LocalIndex* ordering,
                             Vector& buffer)
{
    Vector temp(numAssigned);

    // extract what we already had before the exchange
    gatherCpu(ordering, numPresent, buffer.data() + bufDesc.start, temp.data());

    // extract what we received during the exchange
    LocalIndex rStart = receiveStart(bufDesc, numPresent, numAssigned);
    std::copy_n(buffer.data() + rStart, numAssigned - numPresent, temp.data() + numPresent);
    swap(temp, buffer);
}

/*! @brief Only used in testing to isolate locally owned particles after calling exchangeParticles.
 *         In production code, this step is deferred until after halo detection to rearrange particles to their final
 *         location in a single step.
 */
template<class... Vector>
void extractLocallyOwned(BufferDescription bufDesc,
                         LocalIndex numPresent,
                         LocalIndex numAssigned,
                         const LocalIndex* ordering,
                         Vector&... buffers)
{
    std::initializer_list<int>{(extractLocallyOwnedImpl(bufDesc, numPresent, numAssigned, ordering, buffers), 0)...};
}

} // namespace domain_exchange

} // namespace cstone
