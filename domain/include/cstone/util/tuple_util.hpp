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

template<class Tuple, size_t... Is>
constexpr auto discardLastImpl(const Tuple& tuple, std::index_sequence<Is...>)
{
    return std::tie(std::get<Is>(tuple)...);
}

template<class Tuple>
constexpr auto discardLastElement(const Tuple& tuple)
{
    constexpr int tupleSize = std::tuple_size_v<Tuple>;
    static_assert(tupleSize > 1);

    using Seq = std::make_index_sequence<tupleSize - 1>;
    return discardLastImpl(tuple, Seq{});
}

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

template<class Tuple>
decltype(auto) reverse(Tuple&& tuple)
{
    return selectTuple(std::forward<Tuple>(tuple), makeIndexSequenceReverse<std::tuple_size_v<std::decay_t<Tuple>>>{});
}

} // namespace util
