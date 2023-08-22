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
 * @brief Utility functions to use compile-time strings as arguments to get on tuples
 *
 * Needs C++20 structural types
 */

#pragma once

#include "cstone/util/constexpr_string.hpp"

namespace require_gcc_12
{

//! @brief Base template for a holder of compile-time values
template<auto...>
struct ValueList
{
};

template<class VL>
struct ValueListSize
{
};

template<template<auto...> class VL, auto... Vs>
struct ValueListSize<VL<Vs...>> : public std::integral_constant<std::size_t, sizeof...(Vs)>
{
};

template<size_t I, class VL>
struct ValueListElement
{
};

//! @brief Element type retrieval: recursion, strip one element
template<size_t I, template<auto...> class VL, auto Head, auto... Tail>
struct ValueListElement<I, VL<Head, Tail...>> : public ValueListElement<I - 1, VL<Tail...>>
{
};

//! @brief Element type retrieval: endpoint, Head is the desired type
template<auto Head, template<auto...> class TL, auto... Tail>
struct ValueListElement<0, TL<Head, Tail...>>
{
    constexpr inline static auto value = Head;
};

//! @brief Element type retrieval: out of bounds detection
template<size_t I, template<auto...> class VL>
struct ValueListElement<I, VL<>>
{
    static_assert(I < ValueListSize<VL<>>{}, "ValueListElement access out of range");
};

namespace detail
{

//! @brief Recursion to check the next field N+1
template<int N, auto V, class VL, class Match = void>
struct MatchValue : std::integral_constant<size_t, MatchValue<N + 1, V, VL>{}>
{
};

//! @brief recursion stop when V == ValueListElement<N, VL>::value is true
template<int N, auto V, class VL>
struct MatchValue<N, V, VL, std::enable_if_t<V == ValueListElement<N, VL>::value>> : std::integral_constant<size_t, N>
{
};

} // namespace detail

/*! @brief Meta function to return the first index in VL whose value matches V
 *
 *  If there are more than one, the first occurrence will be returned.
 *  If there is no such type, the size of VL will be returned.
 */
template<auto V, class VL>
struct FindIndex
{
};

/*! @brief Specialization to only enable this trait if TL has non-type template parameters
 *
 * @tparam V          a value to look for in the template parameters of TL
 * @tparam VL         a non-type template template parameter, e.g. an instance of ValueList
 * @tparam Vs         template parameters of VL
 * @tparam Comparison comparison operation
 *
 *  Note that @p V is added to @pa VL as a sentinel to terminate the recursion
 *  and prevent an out of bounds tuple access compiler error.
 */
template<auto V, template<auto...> class VL, auto... Vs>
struct FindIndex<V, VL<Vs...>> : detail::MatchValue<0, V, VL<Vs..., V>>
{
};

} // namespace require_gcc_12

namespace util
{

/*!**************************************
 Simplified ValueList restricted to structural strings to work around internal compiler errors on GCC 11
 This can be removed and replaced by the more general ValueList once GCC 11 support is no longer needed.
***************************************/

namespace vl_detail
{

template<size_t I, bool match, StructuralString SS, StructuralString Head, StructuralString... Tail>
struct MatchField_helper : public MatchField_helper<I + 1, SS == Head, SS, Tail...>
{
};

template<size_t I, StructuralString SS, StructuralString Head, StructuralString... Tail>
struct MatchField_helper<I, true, SS, Head, Tail...> : std::integral_constant<size_t, I>
{
};

template<size_t I, StructuralString SS, class VL>
struct MatchField
{
};

//! @brief Element retrieval: recursion, strip one element
template<size_t I, StructuralString SS, StructuralString Head, StructuralString... Tail>
struct MatchField<I, SS, FieldList<Head, Tail...>> : public MatchField_helper<I, SS == Head, SS, Tail...>
{
};

template<StructuralString V, class VL>
struct FindIndex
{
};

template<StructuralString V, StructuralString... Vs>
struct FindIndex<V, FieldList<Vs...>> : MatchField<0, V, FieldList<Vs..., V>>
{
};

} // namespace vl_detail

/***************************************/

//! @brief Access a field of a tuple with named fields
template<StructuralString F, class FieldNames, class Tuple>
decltype(auto) get(Tuple&& tuple)
{
    return get<vl_detail::FindIndex<F, FieldNames>{}>(std::forward<Tuple>(tuple));
}

namespace vl_detail
{

template<class T, class IntSeq>
struct MakeFieldListHelper
{
};

template<class T, size_t... Is>
struct MakeFieldListHelper<T, std::integer_sequence<size_t, Is...>>
{
    // +1 to accomodate the '\0' character
    using type = util::FieldList<util::StructuralString<std::char_traits<char>::length(T::fieldNames[Is]) + 1>(
        T::fieldNames[Is])...>;
};

} // namespace vl_detail

//! @brief Construct a FieldList type from any type with a constexpr array<N, const char*> fieldNames member
template<class T>
struct MakeFieldList
{
    inline static constexpr int numFields = T::fieldNames.size();
    using Fields = typename vl_detail::MakeFieldListHelper<T, std::make_index_sequence<numFields>>::type;
};

} // namespace util
