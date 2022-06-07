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
 * @brief  Kahan summation
 *
 * @author Rio Yokota <rioyokota@gsic.titech.ac.jp>
 */

#pragma once

#include <iostream>

#include "cstone/cuda/annotation.hpp"

//! Operator overloading for Kahan summation
template<typename T>
struct kahan
{
    T                               s;
    T                               c;
    HOST_DEVICE_FUN __forceinline__ kahan() {} // Default constructor
    HOST_DEVICE_FUN __forceinline__ kahan(const T& v)
    { // Copy constructor (scalar)
        s = v;
        c = 0;
    }
    HOST_DEVICE_FUN kahan(const kahan& v)
    { // Copy constructor (structure)
        s = v.s;
        c = v.c;
    }
    HOST_DEVICE_FUN ~kahan() {} // Destructor
    HOST_DEVICE_FUN const kahan& operator=(const T v)
    { // Scalar assignment
        s = v;
        c = 0;
        return *this;
    }
    HOST_DEVICE_FUN const kahan& operator+=(const T v)
    { // Scalar compound assignment (add)
        T y = v - c;
        T t = s + y;
        c   = (t - s) - y;
        s   = t;
        return *this;
    }
    HOST_DEVICE_FUN const kahan& operator-=(const T v)
    { // Scalar compound assignment (subtract)
        T y = -v - c;
        T t = s + y;
        c   = (t - s) - y;
        s   = t;
        return *this;
    }
    HOST_DEVICE_FUN const kahan& operator*=(const T v)
    { // Scalar compound assignment (multiply)
        c *= v;
        s *= v;
        return *this;
    }
    HOST_DEVICE_FUN const kahan& operator/=(const T v)
    { // Scalar compound assignment (divide)
        c /= v;
        s /= v;
        return *this;
    }
    HOST_DEVICE_FUN const kahan& operator=(const kahan& v)
    { // Vector assignment
        s = v.s;
        c = v.c;
        return *this;
    }
    HOST_DEVICE_FUN const kahan& operator+=(const kahan& v)
    { // Vector compound assignment (add)
        T y = v.s - c;
        T t = s + y;
        c   = (t - s) - y;
        s   = t;
        y   = v.c - c;
        t   = s + y;
        c   = (t - s) - y;
        s   = t;
        return *this;
    }
    HOST_DEVICE_FUN const kahan& operator-=(const kahan& v)
    { // Vector compound assignment (subtract)
        T y = -v.s - c;
        T t = s + y;
        c   = (t - s) - y;
        s   = t;
        y   = -v.c - c;
        t   = s + y;
        c   = (t - s) - y;
        s   = t;
        return *this;
    }
    HOST_DEVICE_FUN const kahan& operator*=(const kahan& v)
    { // Vector compound assignment (multiply)
        c *= (v.c + v.s);
        s *= (v.c + v.s);
        return *this;
    }
    HOST_DEVICE_FUN const kahan& operator/=(const kahan& v)
    { // Vector compound assignment (divide)
        c /= (v.c + v.s);
        s /= (v.c + v.s);
        return *this;
    }
    HOST_DEVICE_FUN kahan operator-() const
    { // Vector arithmetic (negation)
        kahan temp;
        temp.s = -s;
        temp.c = -c;
        return temp;
    }
    HOST_DEVICE_FUN      operator T() { return s + c; }             // Type-casting (lvalue)
    HOST_DEVICE_FUN      operator const T() const { return s + c; } // Type-casting (rvalue)
    friend std::ostream& operator<<(std::ostream& s, const kahan& v)
    { // Output stream
        s << (v.s + v.c);
        return s;
    }
    friend std::istream& operator>>(std::istream& s, kahan& v)
    { // Input stream
        s >> v.s;
        v.c = 0;
        return s;
    }
};
