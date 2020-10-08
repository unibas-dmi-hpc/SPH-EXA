#pragma once

#include <algorithm>
#include <cassert>
#include <numeric>
#include <tuple>
#include <vector>

#include "mortoncode.hpp"

namespace sphexa
{
/*! \brief calculate the sortKey that sorts the input sequence
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
                   const Box<T>&  box)
{
    std::size_t n = std::distance(xBegin, xEnd);

    std::vector<unsigned> mortonCodes(n); 
    computeMortonCodes(xBegin, xEnd, yBegin, zBegin, begin(mortonCodes), box);

    sort_invert(cbegin(mortonCodes), cend(mortonCodes), orderBegin);
}

} // namespace sphexa
