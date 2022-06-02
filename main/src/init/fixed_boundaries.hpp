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
template<class T, class Dataset>
void applyFixedBoundaries(Dataset& d, int fbc[3], cstone::Box<T>& box)
{

    if (fbc[0]) // fixed boundaries in x-direction
    {
#pragma omp parallel for
        for (size_t i = 0; i < d.x.size(); i++)
        {
            T distXmax = std::abs(box.xmax() - d.x[i]);
            T distXmin = std::abs(box.xmin() - d.x[i]);

            if (distXmax < 2.0 * d.h[0] || distXmin < 2.0 * d.h[0])
            {
                d.hasFBC[i] = 1.0;
                d.vx[i]     = 0.0;
                d.vy[i]     = 0.0;
                d.vz[i]     = 0.0;
            }
            else { d.hasFBC[i] = 0.0; }
        }
    }
    if (fbc[1]) // fixed boundaries in y-direction
    {
#pragma omp parallel for
        for (size_t i = 0; i < d.x.size(); i++)
        {
            T distYmax = std::abs(box.ymax() - d.y[i]);
            T distYmin = std::abs(box.ymin() - d.y[i]);

            if (distYmax < 2.0 * d.h[0] || distYmin < 2.0 * d.h[0])
            {
                d.hasFBC[i] = 1.0;
                d.vx[i]     = 0.0;
                d.vy[i]     = 0.0;
                d.vz[i]     = 0.0;
            }
            else { d.hasFBC[i] = 0.0; }
        }
    }
    if (fbc[2]) // fixed boundaries in z-direction
    {
#pragma omp parallel for
        for (size_t i = 0; i < d.x.size(); i++)
        {
            T distZmax = std::abs(box.zmax() - d.z[i]);
            T distZmin = std::abs(box.zmin() - d.z[i]);

            if (distZmax < 2.0 * d.h[0] || distZmin < 2.0 * d.h[0])
            {
                d.hasFBC[i] = 1.0;
                d.vx[i]     = 0.0;
                d.vy[i]     = 0.0;
                d.vz[i]     = 0.0;
            }
            else { d.hasFBC[i] = 0.0; }
        }
    }
}
} // namespace sphexa
