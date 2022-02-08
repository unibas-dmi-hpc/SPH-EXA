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

#include "ryoanji/types.h"

namespace ryoanji
{

template<class T>
static void makeCubeBodies(Vec4<T>* bodies, size_t n, double extent = 3)
{
    for (size_t i = 0; i < n; i++)
    {
        bodies[i][0] = drand48() * 2 * extent - extent;
        bodies[i][1] = drand48() * 2 * extent - extent;
        bodies[i][2] = drand48() * 2 * extent - extent;
        bodies[i][3] = drand48() / n;
    }

    // set non-random corners
    bodies[0][0] = -extent;
    bodies[0][1] = -extent;
    bodies[0][2] = -extent;

    bodies[n - 1][0] = extent;
    bodies[n - 1][1] = extent;
    bodies[n - 1][2] = extent;
}

//! generate a grid with npOnEdge^3 bodies
template<class T>
static void makeGridBodies(Vec4<T>* bodies, int npOnEdge, double spacing)
{
    for (size_t i = 0; i < npOnEdge; i++)
        for (size_t j = 0; j < npOnEdge; j++)
            for (size_t k = 0; k < npOnEdge; k++)
            {
                size_t linIdx     = i * npOnEdge * npOnEdge + j * npOnEdge + k;
                bodies[linIdx][0] = i * spacing;
                bodies[linIdx][1] = j * spacing;
                bodies[linIdx][2] = k * spacing;
                bodies[linIdx][3] = 1.0 / (npOnEdge * npOnEdge * npOnEdge);
            }
}

} // namespace ryoanji