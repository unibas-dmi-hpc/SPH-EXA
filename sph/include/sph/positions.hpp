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

//! @brief Update positions according to Press (2nd order)
template<class T, class Tc, class Ta, class Tm1>
HOST_DEVICE_FUN auto positionUpdate(T dt, T dt_m1, cstone::Vec3<Tc> X, cstone::Vec3<Ta> A, cstone::Vec3<Tm1> X_m1,
                                    const cstone::Box<Tc>& box)
{
    T deltaA = dt + T(0.5) * dt_m1;
    T deltaB = T(0.5) * (dt + dt_m1);

    auto Val = (X - X_m1) * (T(1) / dt_m1);

    auto V = Val + A * deltaA;
    X_m1   = X;
    X += dt * Val + A * deltaB * dt;

    auto Xpbc = cstone::applyPbc(X, box);
    X_m1 += Xpbc - X;
    X = Xpbc;

    return util::tuple<cstone::Vec3<Tc>, cstone::Vec3<Tc>, cstone::Vec3<Tm1>>{X, V, X_m1};
}

template<class T, class Dataset>
void computePositions(size_t startIndex, size_t endIndex, Dataset& d, const cstone::Box<T>& box)
{
    using Vec3T = cstone::Vec3<T>;
    T dt        = d.minDt;
    T dt_m1     = d.minDt_m1;

    const auto* du = d.du.data();
    const auto* h  = d.h.data();

    auto* x     = d.x.data();
    auto* y     = d.y.data();
    auto* z     = d.z.data();
    auto* vx    = d.vx.data();
    auto* vy    = d.vy.data();
    auto* vz    = d.vz.data();
    auto* x_m1  = d.x_m1.data();
    auto* y_m1  = d.y_m1.data();
    auto* z_m1  = d.z_m1.data();
    auto* u     = d.u.data();
    auto* du_m1 = d.du_m1.data();

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
        Vec3T V;
        util::tie(X, V, X_m1) = positionUpdate(dt, dt_m1, X, A, X_m1, box);

        util::tie(x[i], y[i], z[i])          = util::tie(X[0], X[1], X[2]);
        util::tie(x_m1[i], y_m1[i], z_m1[i]) = util::tie(X_m1[0], X_m1[1], X_m1[2]);
        util::tie(vx[i], vy[i], vz[i])       = util::tie(V[0], V[1], V[2]);

        u[i] += energyUpdate(dt, dt_m1, du[i], du_m1[i]);
        du_m1[i] = du[i];
    }
}

} // namespace sph

