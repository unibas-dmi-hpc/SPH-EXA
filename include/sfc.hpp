#pragma once

#include <algorithm>
#include <cassert>
#include <numeric>
#include <tuple>
#include <vector>

namespace sphexa
{

namespace detail {

//! \brief Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
unsigned int expandBits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

template<class T>
static inline T normalize(T d, T min, T max) { return (d - min) / (max - min); }

} // namespace detail

/*! \brief Calculates a 30-bit Morton code for a 3D point
 *
 * \param[in] x,y,z input coordinates within the unit cube [0,1]^3
 */
template<class T>
unsigned int morton3D(T x, T y, T z)
{
    assert( x >= 0.0 && x <= 1.0);
    assert( y >= 0.0 && y <= 1.0);
    assert( z >= 0.0 && z <= 1.0);

    // normalize floating point numbers
    // 1024 = 2^10, so we map the floating point numbers
    // in [0,1] to [0,1023] and convert to integers
    x = std::min(std::max(x * T(1024.0), T(0.0)), T(1023.0));
    y = std::min(std::max(y * T(1024.0), T(0.0)), T(1023.0));
    z = std::min(std::max(z * T(1024.0), T(0.0)), T(1023.0));
    unsigned int xx = detail::expandBits((unsigned int)x);
    unsigned int yy = detail::expandBits((unsigned int)y);
    unsigned int zz = detail::expandBits((unsigned int)z);

    // interleave the x, y, z components
    return xx * 4 + yy * 2 + zz;
}


/*! \brief calculate the sortKey that sortes the input sequence
 *
 * \param[in] inBegin input sequence start 
 * \param[in] inEnd input sequence end
 * \param[out] outBegin integer sortKey output
 *
 * upon completion of this routine, the output sequence containes the sort keys
 * that will access the input sequence in a sorted manner, i.e.
 * the sequence inBegin[outBegin[k]], k=0,1,2,...,n is sorted
 */
template <class InputIterator, class OutputIterator>
void sort_invert(InputIterator inBegin, InputIterator inEnd, OutputIterator outBegin)
{
    using ValueType = std::decay_t<decltype(*inBegin)>;
    using Integer   = std::decay_t<decltype(*outBegin)>;
    std::size_t n   = std::distance(inBegin, inEnd);

    OutputIterator outEnd = outBegin + n;
    // create index sequence 0,1,2,...,n
    std::iota(outBegin, outEnd, 0);

    // zip the input integer array together with the index sequence
    std::vector<std::tuple<ValueType, Integer>> keyIndexPairs(n);
    std::transform(inBegin, inEnd, outBegin, begin(keyIndexPairs),
                   [](ValueType i1, Integer i2) { return std::make_tuple(i1, i2); });

    // sort, comparing only the first tuple element
    std::sort(begin(keyIndexPairs), end(keyIndexPairs),
              [](const auto& t1, const auto& t2){ return std::get<0>(t1) < std::get<0>(t2); });

    // extract the resulting ordering
    std::transform(begin(keyIndexPairs), end(keyIndexPairs), outBegin,
                   [](const auto& tuple) { return std::get<1>(tuple); } );
}

/*! \brief compute the Morton codes for the input coordinate arrays
 *
 * \param[in]  [x,y,z][Begin, End] (const) input iterators for coordinate arrays
 * \param[out] order[Begin, End]  output for morton codes
 * \param[in]  [x,y,z][min, max]  coordinate bounding box
 */
template<class InputIterator, class OutputIterator, class T>
void computeMortonCodes(InputIterator  xBegin,
                        InputIterator  xEnd,
                        InputIterator  yBegin,
                        InputIterator  zBegin,
                        OutputIterator codesBegin,
                        T xmin, T xmax, T ymin, T ymax, T zmin, T zmax)
{
    using detail::normalize; 

    while (xBegin != xEnd)
    {
        *codesBegin++ = morton3D(normalize(*xBegin++, xmin, xmax),
                                 normalize(*yBegin++, ymin, ymax),
                                 normalize(*zBegin++, zmin, zmax));
    }
}


/*! \brief compute the Morton z-order for the input coordinate arrays
 *
 * \param[in]  [x,y,z][Begin, End] (const) input iterators for coordinate arrays
 * \param[out] order[Begin, End]  output for z-order
 * \param[in]  [x,y,z][min, max]  coordinate bounding box
 */
template<class InputIterator, class OutputIterator, class T>
void computeZorder(InputIterator  xBegin,
                   InputIterator  xEnd,
                   InputIterator  yBegin,
                   InputIterator  zBegin,
                   OutputIterator orderBegin,
                   T xmin, T xmax, T ymin, T ymax, T zmin, T zmax)
{
    std::size_t n = std::distance(xBegin, xEnd);

    std::vector<unsigned> mortonCodes(n); 
    computeMortonCodes(xBegin, xEnd, yBegin, zBegin, begin(mortonCodes), xmin, xmax, ymin, ymax, zmin, xmax);

    sort_invert(begin(mortonCodes), end(mortonCodes), orderBegin);
}



} // namespace sphexa
