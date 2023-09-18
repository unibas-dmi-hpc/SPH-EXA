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
 * @brief  Constexpr string based on C++20 structural types
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <algorithm>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace util
{

//! @brief constexpr string as structural type for use as non-type template parameter (C++20)
template<size_t N>
struct StructuralString
{
    //! @brief construction from string literal
    constexpr StructuralString(const char (&str)[N]) noexcept { std::copy_n(str, N, value); }

    //! @brief construction from string_view or const char*, needs explicit specification of template arg N
    constexpr StructuralString(std::string_view str) noexcept { std::copy_n(str.data(), N, value); }

    char value[N];
};

template<size_t N1, size_t N2>
constexpr bool operator==(const StructuralString<N1>& a, const StructuralString<N2>& b)
{
    return (N1 == N2) && std::equal(a.value, a.value + N1, b.value);
}

template<size_t N1, size_t N2>
constexpr StructuralString<N1 + N2 - 1> operator+(const StructuralString<N1>& a, const StructuralString<N2>& b)
{
    char value[N1 + N2 - 1];
    std::copy_n(a.value, N1 - 1, value);
    std::copy_n(b.value, N2, value + N1 - 1);
    return StructuralString(value);
}

template<StructuralString... Fields>
struct FieldList
{
};

template<StructuralString... F1, StructuralString... F2>
constexpr auto operator+(FieldList<F1...>, FieldList<F2...>)
{
    return FieldList<F1..., F2...>{};
}

template<class VL>
struct FieldListSize
{
};

template<StructuralString... Vs>
struct FieldListSize<FieldList<Vs...>> : public std::integral_constant<std::size_t, sizeof...(Vs)>
{
};

template<StructuralString... Fields>
constexpr auto make_tuple(FieldList<Fields...>)
{
    return std::make_tuple(Fields...);
}

} // namespace util
