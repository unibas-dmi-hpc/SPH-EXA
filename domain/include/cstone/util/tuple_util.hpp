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
 * @brief  General purpose utilities
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <algorithm>
#include <tuple>
#include <type_traits>
#include <utility>

namespace util
{

//! @brief Utility to call function with each element in tuple_
template<class F, class Tuple>
void for_each_tuple(F&& func, Tuple&& tuple_)
{
    std::apply([f = func](auto&&... args) { [[maybe_unused]] auto list = std::initializer_list<int>{(f(args), 0)...}; },
               std::forward<Tuple>(tuple_));
}

//! @brief convert an index_sequence into a tuple of integral constants (e.g. for use with for_each_tuple)
template<size_t... Is>
constexpr auto makeIntegralTuple(std::index_sequence<Is...>)
{
    return std::make_tuple(std::integral_constant<size_t, Is>{}...);
}

//! @brief Select tuple elements specified by the argument sequence
template<class Tuple, std::size_t... Ints>
std::tuple<std::tuple_element_t<Ints, std::decay_t<Tuple>>...> selectTuple(Tuple&& tuple, std::index_sequence<Ints...>)
{
    return {std::get<Ints>(std::forward<Tuple>(tuple))...};
}

template<std::size_t... Is>
constexpr auto indexSequenceReverse(std::index_sequence<Is...> const&)
    -> decltype(std::index_sequence<sizeof...(Is) - 1U - Is...>{});

template<std::size_t N>
using makeIndexSequenceReverse = decltype(indexSequenceReverse(std::make_index_sequence<N>{}));

//! @brief Create a new tuple by reversing the element order of the argument tuple
template<class Tuple>
decltype(auto) reverse(Tuple&& tuple)
{
    return selectTuple(std::forward<Tuple>(tuple), makeIndexSequenceReverse<std::tuple_size_v<std::decay_t<Tuple>>>{});
}

//! @brief Return a new tuple without the last element of the argument tuple
template<class Tp>
constexpr auto discardLastElement(Tp&& tp)
{
    return selectTuple(std::forward<Tp>(tp), std::make_index_sequence<std::tuple_size_v<std::decay_t<Tp>> - 1>{});
}

/*! @brief Zip multiple tuples into a single tuple, similar to C++23 std::views::zip, but for tuples (no iterators)
 *
 * @tparam Tps types of tuples
 * @param tps  some tuples, tuple(A0, ..., An), tuple(B0, ..., Bn)
 * @return     a single  tuple( tuple(A0, B0, ...), ...)
 */
template<class... Tps>
constexpr auto zipTuples(Tps&&... tps)
{
    constexpr std::size_t N = std::min({std::tuple_size_v<std::decay_t<Tps>>...});

    // auto zip = [&tps...]<std::size_t... Is>(std::index_sequence<Is...>) // C++20 (not supported by CUDA 11)
    auto zip = [&tps...](auto... Is)
    {
        auto getIs = [](auto I, Tps&&... tps)
        { return std::tuple<std::tuple_element_t<I, std::decay_t<Tps>>...>{std::get<I>(std::forward<Tps>(tps))...}; };
        return std::make_tuple(getIs(std::integral_constant<size_t, Is>{}, std::forward<Tps>(tps)...)...);
    };

    return std::apply(zip, makeIntegralTuple(std::make_index_sequence<N>{})); // zip(std::make_index_sequence<N>{})
}

} // namespace util
