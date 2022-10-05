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
#include <utility>

#include "cstone/cuda/annotation.hpp"
#include "array.hpp"

/*! @brief A template to create structs as a type-safe version to using declarations
 *
 * Used in public API functions where a distinction between different
 * arguments of the same underlying type is desired. This provides a type-safe
 * version to using declarations. Instead of naming a type alias, the name
 * is used to define a struct that inherits from StrongType<T>, where T is
 * the underlying type.
 *
 * Due to the T() conversion and assignment from T,
 * an instance of StrongType<T> struct behaves essentially like an actual T, while construction
 * from T is disabled. This makes it impossible to pass a T as a function parameter
 * of type StrongType<T>.
 */
template<class T, class Phantom>
struct StrongType
{
    using ValueType [[maybe_unused]] = T;

    //! default ctor
    constexpr HOST_DEVICE_FUN StrongType()
        : value_{}
    {
    }
    //! construction from the underlying type T, implicit conversions disabled
    explicit constexpr HOST_DEVICE_FUN StrongType(T v)
        : value_(std::move(v))
    {
    }

    //! assignment from T
    constexpr HOST_DEVICE_FUN StrongType& operator=(T v)
    {
        value_ = std::move(v);
        return *this;
    }

    //! conversion to T
    constexpr HOST_DEVICE_FUN operator T() const { return value_; } // NOLINT

    //! access the underlying value
    constexpr HOST_DEVICE_FUN T value() const { return value_; }

private:
    T value_;
};

/*! @brief StrongType equality comparison
 *
 * Requires that both T and Phantom template parameters match.
 * For the case where a comparison between StrongTypes with matching T, but differing Phantom
 * parameters is desired, the underlying value attribute should be compared instead
 */
template<class T, class Phantom>
constexpr HOST_DEVICE_FUN bool operator==(const StrongType<T, Phantom>& lhs, const StrongType<T, Phantom>& rhs)
{
    return lhs.value() == rhs.value();
}

//! @brief comparison function <
template<class T, class Phantom>
constexpr HOST_DEVICE_FUN bool operator<(const StrongType<T, Phantom>& lhs, const StrongType<T, Phantom>& rhs)
{
    return lhs.value() < rhs.value();
}

//! @brief comparison function >
template<class T, class Phantom>
constexpr HOST_DEVICE_FUN bool operator>(const StrongType<T, Phantom>& lhs, const StrongType<T, Phantom>& rhs)
{
    return lhs.value() > rhs.value();
}

//! @brief addition
template<class T, class Phantom>
constexpr HOST_DEVICE_FUN StrongType<T, Phantom> operator+(const StrongType<T, Phantom>& lhs,
                                                           const StrongType<T, Phantom>& rhs)
{
    return StrongType<T, Phantom>(lhs.value() + rhs.value());
}

//! @brief subtraction
template<class T, class Phantom>
constexpr HOST_DEVICE_FUN StrongType<T, Phantom> operator-(const StrongType<T, Phantom>& lhs,
                                                           const StrongType<T, Phantom>& rhs)
{
    return StrongType<T, Phantom>(lhs.value() - rhs.value());
}

//! @brief constexpr string as structural type for use as non-type template parameter (C++20)
template<size_t N>
struct StructuralString
{
    constexpr StructuralString(const char (&str)[N]) noexcept { std::copy_n(str, N, value); }

    char value[N];
};

template<size_t N1, size_t N2>
constexpr StructuralString<N1 + N2 - 1> operator+(const StructuralString<N1>& a, const StructuralString<N2>& b)
{
    char value[N1 + N2 - 1];
    std::copy_n(a.value, N1 - 1, value);
    std::copy_n(b.value, N2, value + N1 - 1);
    return StructuralString(value);
}

//! @brief Utility to call function with each element in tuple_
template<class F, class... Ts>
void for_each_tuple(F&& func, std::tuple<Ts...>& tuple_)
{
    std::apply([f = func](auto&... args) { [[maybe_unused]] auto list = std::initializer_list<int>{(f(args), 0)...}; },
               tuple_);
}

//! @brief Utility to call function with each element in tuple_ with const guarantee
template<class F, class... Ts>
void for_each_tuple(F&& func, const std::tuple<Ts...>& tuple_)
{
    std::apply([f = func](auto&... args) { [[maybe_unused]] auto list = std::initializer_list<int>{(f(args), 0)...}; },
               tuple_);
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

//! @brief ceil(dividend/divisor) for integers
HOST_DEVICE_FUN constexpr unsigned iceil(size_t dividend, unsigned divisor)
{
    return (dividend + divisor - 1) / divisor;
}

HOST_DEVICE_FUN constexpr size_t round_up(size_t n, unsigned multiple) { return iceil(n, multiple) * multiple; }
