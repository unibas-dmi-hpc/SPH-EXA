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

/*! \file
 * \brief Functionality for calculating the z-order and for reordering arrays according to it.
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 */


#pragma once

#include <algorithm>
#include <cassert>
#include <numeric>
#include <tuple>
#include <vector>

#include "mortoncode.hpp"

namespace cstone
{

/*! \brief calculate the sortKey that sorts the input sequence
 *
 * \param[in] inBegin input sequence start 
 * \param[in] inEnd input sequence end
 * \param[out] outBegin integer sortKey output
 *
 * upon completion of this routine, the output sequence contains the sort keys
 * that will access the input sequence in a sorted manner, i.e.
 * the sequence inBegin[outBegin[k]], k=0,1,2,...,n is sorted
 */
template <class InputIterator, class OutputIterator>
void sort_invert(InputIterator inBegin, InputIterator inEnd, OutputIterator outBegin)
{
    using ValueType = std::decay_t<decltype(*inBegin)>;
    using Integer   = std::decay_t<decltype(*outBegin)>;
    std::size_t n   = std::distance(inBegin, inEnd);

    // create index sequence 0,1,2,...,n
    #pragma omp parallel for
    for (std::size_t i = 0; i < n; ++i)
        outBegin[i] = i;

    // zip the input integer array together with the index sequence
    std::vector<std::tuple<ValueType, Integer>> keyIndexPairs(n);
    #pragma omp parallel for
    for (std::size_t i = 0; i < n; ++i)
        keyIndexPairs[i] = std::make_tuple(inBegin[i], outBegin[i]);

    // sort, comparing only the first tuple element
    std::sort(begin(keyIndexPairs), end(keyIndexPairs),
              [](const auto& t1, const auto& t2){ return std::get<0>(t1) < std::get<0>(t2); });

    // extract the resulting ordering
    #pragma omp parallel for
    for (std::size_t i = 0; i < n; ++i)
        outBegin[i] = std::get<1>(keyIndexPairs[i]);
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


/*! \brief reorder the input array according to the specified ordering
 *
 * \tparam I          integer type
 * \tparam ValueType  float or double
 * \param ordering    an ordering
 * \param array       an array, size >= ordering.size(), particles past ordering.size()
 *                    are copied element by element
 */
template<class I, class ValueType>
void reorder(const std::vector<I>& ordering, std::vector<ValueType>& array)
{
    assert(array.size() >= ordering.size());

    std::vector<ValueType> tmp(array.size());
    for (std::size_t i = 0; i < ordering.size(); ++i)
    {
        tmp[i] = array[ordering[i]];
    }
    for (std::size_t i = ordering.size(); i < array.size(); ++i)
    {
        tmp[i] = array[i];
    }
    std::swap(tmp, array);
}

/*! \brief reorder the input array according to the specified ordering
 *
 * \tparam I          integer type
 * \tparam ValueType  float or double
 * \param ordering    an ordering, all indices from 0 to ordering.size() are accessed
 * \param array       an array, indices offset to offset + ordering.size() are reordered.
 *                    other elements are copied element by element
 * \param offset      access array with an offset
 */
template<class I, class ValueType>
void reorder(const std::vector<I>& ordering, std::vector<ValueType>& array, int offset)
{
    assert(array.size() >= ordering.size() + offset);

    std::vector<ValueType> tmp(array.size());

    for (int i = 0; i < offset; ++i)
    {
        tmp[i] = array[i];
    }

    for (std::size_t i = 0; i < ordering.size(); ++i)
    {
        tmp[i+offset] = array[ordering[i]+offset];
    }

    for (size_t i = ordering.size()+offset; i < array.size(); ++i)
    {
        tmp[i] = array[i];
    }

    std::swap(tmp, array);
}

} // namespace cstone
