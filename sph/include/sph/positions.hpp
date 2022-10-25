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

#include "cstone/sfc/box.hpp"
#include "cstone/util/array.hpp"
#include "cstone/util/tuple.hpp"
#include "cstone/tree/accel_switch.hpp"

#include "sph/sph_gpu.hpp"

namespace sph
{

//! @brief checks whether a particle is in the fixed boundary region in one dimension
template<class Tc, class Th>
HOST_DEVICE_FUN bool fbcCheck(Tc coord, Th h, Tc top, Tc bottom, bool fbc)
{
    return fbc && (std::abs(top - coord) < Th(2) * h || std::abs(bottom - coord) < Th(2) * h);
}

//! @brief update the energy according to Adams-Bashforth (2nd order)
template<class T>
HOST_DEVICE_FUN double energyUpdate(double dt, double dt_m1, T du, T du_m1)
{
    double deltaA = 0.5 * dt * dt / dt_m1;
    double deltaB = dt + deltaA;

    return du * deltaB - du_m1 * deltaA;
}

//! @brief Update positions according to Press (2nd order)
template<class T>
HOST_DEVICE_FUN auto positionUpdate(double dt, double dt_m1, cstone::Vec3<T> X, cstone::Vec3<T> A, cstone::Vec3<T> X_m1,
                                    const cstone::Box<T>& box)
{
    double deltaA = dt + T(0.5) * dt_m1;
    double deltaB = T(0.5) * (dt + dt_m1);

    auto Val = X_m1 * (T(1) / dt_m1);
    auto V   = Val + A * deltaA;
    auto dX  = dt * Val + A * deltaB * dt;
    X        = cstone::putInBox(X + dX, box);

    return util::tuple<cstone::Vec3<T>, cstone::Vec3<T>, cstone::Vec3<T>>{X, V, dX};
}

template<class T, class Dataset>
void computePositionsHost(size_t startIndex, size_t endIndex, Dataset& d, const cstone::Box<T>& box)
{
    double dt    = d.minDt;
    double dt_m1 = d.minDt_m1;

    bool fbcX = (box.boundaryX() == cstone::BoundaryType::fixed);
    bool fbcY = (box.boundaryY() == cstone::BoundaryType::fixed);
    bool fbcZ = (box.boundaryZ() == cstone::BoundaryType::fixed);

    bool anyFBC = fbcX || fbcY || fbcZ;

    bool haveMui = !d.mui.empty();
    T    constCv = idealGasCv(d.muiConst);

#pragma omp parallel for schedule(static)
    for (size_t i = startIndex; i < endIndex; i++)
    {
        if (anyFBC && d.vx[i] == T(0) && d.vy[i] == T(0) && d.vz[i] == T(0))
        {
            if (fbcCheck(d.x[i], d.h[i], box.xmax(), box.xmin(), fbcX) ||
                fbcCheck(d.y[i], d.h[i], box.ymax(), box.ymin(), fbcY) ||
                fbcCheck(d.z[i], d.h[i], box.zmax(), box.zmin(), fbcZ))
            {
                continue;
            }
        }

        cstone::Vec3<T> A{d.ax[i], d.ay[i], d.az[i]};
        cstone::Vec3<T> X{d.x[i], d.y[i], d.z[i]};
        cstone::Vec3<T> X_m1{d.x_m1[i], d.y_m1[i], d.z_m1[i]};
        cstone::Vec3<T> V;
        util::tie(X, V, X_m1) = positionUpdate(dt, dt_m1, X, A, X_m1, box);

        util::tie(d.x[i], d.y[i], d.z[i])          = util::tie(X[0], X[1], X[2]);
        util::tie(d.x_m1[i], d.y_m1[i], d.z_m1[i]) = util::tie(X_m1[0], X_m1[1], X_m1[2]);
        util::tie(d.vx[i], d.vy[i], d.vz[i])       = util::tie(V[0], V[1], V[2]);

        T cv = haveMui ? idealGasCv(d.mui[i]) : constCv;
        d.temp[i] += energyUpdate(dt, dt_m1, d.du[i], d.du_m1[i]) / cv;
        d.du_m1[i] = d.du[i];
    }
}

template<class T, class Dataset>
void computePositions(size_t startIndex, size_t endIndex, Dataset& d, const cstone::Box<T>& box)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        T     constCv = d.mui.empty() ? idealGasCv(d.muiConst) : -1.0;
        auto* d_mui   = d.mui.empty() ? nullptr : rawPtr(d.devData.mui);

        computePositionsGpu(startIndex, endIndex, d.minDt, d.minDt_m1, rawPtr(d.devData.x), rawPtr(d.devData.y),
                            rawPtr(d.devData.z), rawPtr(d.devData.vx), rawPtr(d.devData.vy), rawPtr(d.devData.vz),
                            rawPtr(d.devData.x_m1), rawPtr(d.devData.y_m1), rawPtr(d.devData.z_m1),
                            rawPtr(d.devData.ax), rawPtr(d.devData.ay), rawPtr(d.devData.az), rawPtr(d.devData.temp),
                            rawPtr(d.devData.du), rawPtr(d.devData.du_m1), rawPtr(d.devData.h), d_mui, constCv, box);
    }
    else { computePositionsHost(startIndex, endIndex, d, box); }
}

} // namespace sph
