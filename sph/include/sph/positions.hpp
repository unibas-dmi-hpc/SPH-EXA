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

namespace sph
{

//! @brief checks whether a particle is in the fixed boundary region in one dimension
template<class Tc, class Th>
HOST_DEVICE_FUN bool fbcCheck(Tc coord, Th h, Tc top, Tc bottom, bool fbc)
{
    return fbc && (std::abs(top - coord) < Th(2) * h || std::abs(bottom - coord) < Th(2) * h);
}

//! @brief update the energy according to Adams-Bashforth (2nd order)
template<class T1, class T2>
HOST_DEVICE_FUN T2 energyUpdate(T1 dt, T1 dt_m1, T2 du, T2 du_m1)
{
    T1 deltaA = 0.5 * dt * dt / dt_m1;
    T1 deltaB = dt + deltaA;

    return du * deltaB - du_m1 * deltaA;
}

template<class T, class Dataset>
void computePositions(size_t startIndex, size_t endIndex, Dataset& d, const cstone::Box<T>& box)
{
    using Vec3T = cstone::Vec3<T>;
    T dt        = d.minDt;
    T dt_m1     = d.minDt_m1;

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
    T* h     = d.h.data();

    bool pbcX = (box.boundaryX() == cstone::BoundaryType::periodic);
    bool pbcY = (box.boundaryY() == cstone::BoundaryType::periodic);
    bool pbcZ = (box.boundaryZ() == cstone::BoundaryType::periodic);

    bool fbcX = (box.boundaryX() == cstone::BoundaryType::fixed);
    bool fbcY = (box.boundaryY() == cstone::BoundaryType::fixed);
    bool fbcZ = (box.boundaryZ() == cstone::BoundaryType::fixed);

    bool anyFBC = fbcX || fbcY || fbcZ;

#pragma omp parallel for schedule(static)
    for (size_t i = startIndex; i < endIndex; i++)
    {
        if (anyFBC && vx[i] == T(0) && vy[i] == T(0) && vz[i] == T(0))
        {
            if (fbcCheck(x[i], h[i], box.xmax(), box.xmin(), fbcX) ||
                fbcCheck(y[i], h[i], box.ymax(), box.ymin(), fbcY) ||
                fbcCheck(z[i], h[i], box.zmax(), box.zmin(), fbcZ))
            {
                continue;
            }
        }

        Vec3T A{d.ax[i], d.ay[i], d.az[i]};
        Vec3T X{x[i], y[i], z[i]};
        Vec3T X_m1{x_m1[i], y_m1[i], z_m1[i]};

        // Update positions according to Press (2nd order)
        T deltaA = dt + T(0.5) * dt_m1;
        T deltaB = T(0.5) * (dt + dt_m1);

        Vec3T Val = (X - X_m1) * (T(1) / dt_m1);

        Vec3T V = Val + A * deltaA;
        X_m1    = X;
        X += dt * Val + A * deltaB * dt;

        Vec3T Xpbc = cstone::applyPbc(X, box);
        X_m1 += Xpbc - X;
        X = Xpbc;

        util::tie(x[i], y[i], z[i])          = util::tie(X[0], X[1], X[2]);
        util::tie(x_m1[i], y_m1[i], z_m1[i]) = util::tie(X_m1[0], X_m1[1], X_m1[2]);
        util::tie(vx[i], vy[i], vz[i])       = util::tie(V[0], V[1], V[2]);

        u[i] += energyUpdate(dt, dt_m1, du[i], du_m1[i]);
        du_m1[i] = du[i];
    }
}

} // namespace sph

