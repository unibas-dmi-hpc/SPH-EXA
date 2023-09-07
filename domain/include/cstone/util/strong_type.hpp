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

#include "cstone/cuda/annotation.hpp"

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
