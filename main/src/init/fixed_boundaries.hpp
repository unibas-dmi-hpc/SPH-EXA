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

/*! @file flag particles for fixed boundary conditions
 *
 * @author Lukas Schmidt
 */

#pragma once

#include "cstone/sfc/box.hpp"
#include "sph/particles_data.hpp"

namespace sphexa
{
template<class T>
void applyFixedBoundaries(T* pos, T* vx, T* vy, T* vz, T* h, cstone::Box<T>& box, size_t first, size_t last)
{

#pragma omp parallel for
    for (size_t i = first; i < last; i++)
    {
        T distMax = std::abs(box.xmax() - pos[i]);
        T distMin = std::abs(box.xmin() - pos[i]);

        if (distMax < 2.0 * h[i] || distMin < 2.0 * h[i])
        {
            vx[i] = 0.0;
            vy[i] = 0.0;
            vz[i] = 0.0;
        }
    }
}

} // namespace sphexa
