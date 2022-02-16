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
 * @brief 2nd order time-step integrator
 *
 * @author Aurelien Cavelan
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <vector>
#include <cmath>
#include <tuple>

namespace sphexa
{
namespace sph
{

template<typename T, class Dataset>
struct computeAccelerationWithGravity
{
    std::tuple<T, T, T> operator()(const int idx, Dataset& d)
    {
        const T G  = d.g;
        const T ax = -(d.grad_P_x[idx] - G * d.fx[idx]);
        const T ay = -(d.grad_P_y[idx] - G * d.fy[idx]);
        const T az = -(d.grad_P_z[idx] - G * d.fz[idx]);
        return std::make_tuple(ax, ay, az);
    }
};

template<class T, class Dataset>
void computePositions(size_t startIndex, size_t endIndex, Dataset& d, const cstone::Box<T>& box)
{
    const T* dt = d.dt.data();
    const T* du = d.du.data();

    T* x     = d.x.data();
    T* y     = d.y.data();
    T* z     = d.z.data();
    T* vx    = d.vx.data();
    T* vy    = d.vy.data();
    T* vz    = d.vz.data();
    T* x_m1  = d.x_m1.data();
    T* y_m1  = d.y_m1.data();
    T* z_m1  = d.z_m1.data();
    T* u     = d.u.data();
    T* du_m1 = d.du_m1.data();
    T* dt_m1 = d.dt_m1.data();

#pragma omp parallel for schedule(static)
    for (size_t i = startIndex; i < endIndex; i++)
    {
        T ax = -d.grad_P_x[i];
        T ay = -d.grad_P_y[i];
        T az = -d.grad_P_z[i];

        // Update positions according to Press (2nd order)
        T deltaA = dt[i] + 0.5 * dt_m1[i];
        T deltaB = 0.5 * (dt[i] + dt_m1[i]);

        T valx = (x[i] - x_m1[i]) / dt_m1[i];
        T valy = (y[i] - y_m1[i]) / dt_m1[i];
        T valz = (z[i] - z_m1[i]) / dt_m1[i];

#ifndef NDEBUG
        if (std::isnan(ax) || std::isnan(ay) || std::isnan(az))
        {
            printf("ERROR::UpdateQuantities(%d) acceleration: (%f %f %f)\n", i, ax, ay, az);
        }
        if (std::isnan(valx) || std::isnan(valy) || std::isnan(valz))
        {
            printf("ERROR::UpdateQuantities(%d) velocities: (%f %f %f)\n", i, valx, valy, valz);
            printf("ERROR::UpdateQuantities(%d) x, y, z, dt_m1: (%f %f %f %f)\n", i, x[i], y[i], z[i], dt_m1[i]);
        }
#endif

        vx[i] = valx + ax * deltaA;
        vy[i] = valy + ay * deltaA;
        vz[i] = valz + az * deltaA;

        x_m1[i] = x[i];
        y_m1[i] = y[i];
        z_m1[i] = z[i];

        x[i] += dt[i] * valx + (vx[i] - valx) * dt[i] * deltaB / deltaA;
        y[i] += dt[i] * valy + (vy[i] - valy) * dt[i] * deltaB / deltaA;
        z[i] += dt[i] * valz + (vz[i] - valz) * dt[i] * deltaB / deltaA;

        if (box.pbcX() && x[i] < box.xmin())
        {
            x[i] += box.lx();
            x_m1[i] += box.lx();
        }
        else if (box.pbcX() && x[i] > box.xmax())
        {
            x[i] -= box.lx();
            x_m1[i] -= box.lx();
        }
        if (box.pbcY() && y[i] < box.ymin())
        {
            y[i] += box.ly();
            y_m1[i] += box.ly();
        }
        else if (box.pbcY() && y[i] > box.ymax())
        {
            y[i] -= box.ly();
            y_m1[i] -= box.ly();
        }
        if (box.pbcZ() && z[i] < box.zmin())
        {
            z[i] += box.lz();
            z_m1[i] += box.lz();
        }
        else if (box.pbcZ() && z[i] > box.zmax())
        {
            z[i] -= box.lz();
            z_m1[i] -= box.lz();
        }

        // Update the energy according to Adams-Bashforth (2nd order)
        deltaA = 0.5 * dt[i] * dt[i] / dt_m1[i];
        deltaB = dt[i] + deltaA;

        u[i] += du[i] * deltaB - du_m1[i] * deltaA;

#ifndef NDEBUG
        if (std::isnan(u[i]) || u[i] < 0.0)
            printf("ERROR::UpdateQuantities(%d) internal energy: u %f du %f dB %f du_m1 %f dA %f\n",
                   i,
                   u[i],
                   du[i],
                   deltaB,
                   du_m1[i],
                   deltaA);
#endif

        du_m1[i] = du[i];
        dt_m1[i] = dt[i];
    }
}

} // namespace sph
} // namespace sphexa
