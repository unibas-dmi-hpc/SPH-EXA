/*! @file
 * @brief  Utility for GPU-direct halo particle exchange
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

namespace cstone
{

template<class T, class IndexType>
extern void gatherRanges(const IndexType* rangeScan,
                         const IndexType* rangeOffsets,
                         int numRanges,
                         const T* src,
                         T* buffer,
                         size_t bufferSize);

} // namespace cstone
