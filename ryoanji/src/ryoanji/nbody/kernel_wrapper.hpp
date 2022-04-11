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
 * @brief compatibility wrappers for Ryoanji EXA-FMM spherical multipoles
 *
 * @author Sebastian Keller        <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <cmath>

#include "cstone/util/tuple.hpp"

#include "cartesian_qpole.hpp"
#include "kernel.hpp"

namespace ryoanji
{

/*! @brief Compute the monopole and quadruple moments from particle coordinates
 */
template<class Tc, class Tm, class Tf, class MType, std::enable_if_t<IsSpherical<MType>{}, int> = 0>
HOST_DEVICE_FUN void particle2Multipole(const Tc* x, const Tc* y, const Tc* z, const Tm* m, LocalIndex first,
                                        LocalIndex last, const Vec3<Tf>& center, MType& Mout)
{
    constexpr int P = ExpansionOrder<MType{}.size()>{};

    Mout = 0;
    for (LocalIndex i = first; i < last; i++)
    {
        Vec3<Tc> body{x[i], y[i], z[i]};
        Vec3<Tc> dX = center - body;

        MType M;
        M[0] = m[i];
        Kernels<0, 0, P - 1>::P2M(M, dX);
        Mout += M;
    }
}

/*! @brief apply gravitational interaction with a multipole to a particle
 *
 * Makes Spherical multipoles usable in the CPU gravity tree traversal implementation
 */
template<class T1, class T2, class MType, std::enable_if_t<IsSpherical<MType>{}, int> = 0>
HOST_DEVICE_FUN DEVICE_INLINE util::tuple<T1, T1, T1, T1> multipole2Particle(T1 tx, T1 ty, T1 tz,
                                                                             const Vec3<T1>& center, MType& M)
{
    Vec3<T1> body{tx, ty, tz};
    Vec4<T1> acc{0, 0, 0, 0};

    acc = M2P(acc, body, center, M, T1(0));
    return {acc[1], acc[2], acc[3], acc[0]};
}

/*! @brief apply gravitational interaction with a multipole to a particle
 *
 * Makes Cartesian multipoles usable in the GPU gravity tree traversal implementation
 */
template<class Ta, class Tc, class MType, std::enable_if_t<IsCartesian<MType>{}, int> = 0>
HOST_DEVICE_FUN DEVICE_INLINE Vec4<Ta> M2P(Vec4<Ta> acc, const Vec3<Tc>& pos_i, const Vec3<Tc>& pos_j, MType& M)
{
    auto [ax, ay, az, p] = multipole2Particle(pos_i[0], pos_i[1], pos_i[2], pos_j, M);
    return {acc[0] + p, acc[1] + ax, acc[2] + ay, acc[3] + az};
}

/*! @brief Combine multipoles into a single multipole
 */
template<class T, class MType, std::enable_if_t<IsSpherical<MType>{}, int> = 0>
HOST_DEVICE_FUN void multipole2Multipole(int begin, int end, const Vec4<T>& Xout, const Vec4<T>* Xsrc,
                                         const MType* Msrc, MType& Mout)
{
    Mout = 0;
    M2M(begin, end, Xout, Xsrc, Msrc, Mout);
}

} // namespace ryoanji
