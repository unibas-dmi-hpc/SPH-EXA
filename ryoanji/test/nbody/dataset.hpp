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
 * @brief Generation of test input bodies
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "ryoanji/nbody/types.h"

namespace ryoanji
{

template<class T>
static void makeCubeBodies(T* x, T* y, T* z, T* m, T* h, size_t n, double extent = 3)
{
    double ng0   = 100;
    T      hInit = std::cbrt(ng0 / n / 4.19) * extent;
    for (size_t i = 0; i < n; i++)
    {
        x[i] = drand48() * 2 * extent - extent;
        y[i] = drand48() * 2 * extent - extent;
        z[i] = drand48() * 2 * extent - extent;
        m[i] = drand48() / n;
        h[i] = hInit;
    }

    // set non-random corners
    x[0] = -extent;
    y[0] = -extent;
    z[0] = -extent;

    x[n - 1] = extent;
    y[n - 1] = extent;
    z[n - 1] = extent;
}

} // namespace ryoanji
