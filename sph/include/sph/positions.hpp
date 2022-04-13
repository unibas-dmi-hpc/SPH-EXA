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
    using Vec3T = cstone::Vec3<T>;
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

    T hasFBC;

#pragma omp parallel for schedule(static)
    for (size_t i = startIndex; i < endIndex; i++)
    {
        if(d.hasFBC[i] == 1.0)
        {
            hasFBC = 0.0;
        } else {
            hasFBC = 1.0;
        }
        Vec3T A{-d.grad_P_x[i], -d.grad_P_y[i], -d.grad_P_z[i]};
        Vec3T X{x[i], y[i], z[i]};
        Vec3T X_m1{x_m1[i], y_m1[i], z_m1[i]};

        T dt_i = dt[i];

        // Update positions according to Press (2nd order)
        T deltaA = dt_i + 0.5 * dt_m1[i];
        T deltaB = 0.5 * (dt_i + dt_m1[i]);

        Vec3T Val = (X - X_m1) * (1.0 * hasFBC/ dt_m1[i]);
        if(Val[0] == 0 && Val[1] == 0 && Val[2] == 0)
        {
            //printf("val zero: i %lu hasFBC %d", i, d.hasFBC[i]);
        }

#ifndef NDEBUG
        if (std::isnan(A[0]) || std::isnan(A[1]) || std::isnan(A[2]))
        {
            printf("ERROR::UpdateQuantities(%lu) acceleration: (%f %f %f)\n", i, A[0], A[1], A[2]);
        }
#endif

        Vec3T V = Val + A * (deltaA * hasFBC);
        X_m1    = X;
        X += (dt_i * hasFBC) * Val + A * deltaB * (dt_i * hasFBC);

        if (box.pbcX() && X[0] < box.xmin())
        {
            X[0] += box.lx();
            X_m1[0] += box.lx();
        }
        else if (box.pbcX() && X[0] > box.xmax())
        {
            X[0] -= box.lx();
            X_m1[0] -= box.lx();
        }
        if (box.pbcY() && X[1] < box.ymin())
        {
            X[1] += box.ly();
            X_m1[1] += box.ly();
        }
        else if (box.pbcY() && X[1] > box.ymax())
        {
            X[1] -= box.ly();
            X_m1[1] -= box.ly();
        }
        if (box.pbcZ() && X[2] < box.zmin())
        {
            X[2] += box.lz();
            X_m1[2] += box.lz();
        }
        else if (box.pbcZ() && X[2] > box.zmax())
        {
            X[2] -= box.lz();
            X_m1[2] -= box.lz();
        }

        x[i]    = X[0];
        y[i]    = X[1];
        z[i]    = X[2];
        x_m1[i] = X_m1[0];
        y_m1[i] = X_m1[1];
        z_m1[i] = X_m1[2];
        vx[i]   = V[0];
        vy[i]   = V[1];
        vz[i]   = V[2];

        // Update the energy according to Adams-Bashforth (2nd order)
        deltaA = 0.5 * dt_i * dt_i / dt_m1[i];
        deltaB = dt_i + deltaA;

        u[i] += du[i] * deltaB - du_m1[i] * deltaA;

        du_m1[i] = du[i];
        dt_m1[i] = dt_i;

#ifndef NDEBUG
        if (std::isnan(u[i]) || u[i] < 0.0)
            printf("ERROR::UpdateQuantities(%lu) internal energy: u %f du %f dB %f du_m1 %f dA %f\n",
                   i,
                   u[i],
                   du[i],
                   deltaB,
                   du_m1[i],
                   deltaA);
#endif
    }
}

} // namespace sph
} // namespace sphexa
