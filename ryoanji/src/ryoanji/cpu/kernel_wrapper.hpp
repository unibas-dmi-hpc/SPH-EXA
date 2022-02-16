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

#include "ryoanji/kernel.hpp"
#include "cstone/util/tuple.hpp"

namespace ryoanji
{

/*! @brief Compute the monopole and quadruple moments from particle coordinates
 */
template<class T1, class T2, class T3>
HOST_DEVICE_FUN void particle2Multipole(const T1* x, const T1* y, const T1* z, const T2* m, LocalIndex first,
                                        LocalIndex last, const Vec3<T1>& center, SphericalMultipole<T3, 2>& Mout)
{
    constexpr int P = 2;

    Mout = 0;
    for (LocalIndex i = first; i < last; i++)
    {
        Vec3<T1> body{x[i], y[i], z[i]};
        Vec3<T1> dX = center - body;

        SphericalMultipole<T3, P> M;
        M[0] = m[i];
        Kernels<0, 0, P - 1>::P2M(M, dX);
        Mout += M;
    }
}

template<class T1, class T2, class T3>
HOST_DEVICE_FUN void particle2Multipole(const T1* x, const T1* y, const T1* z, const T2* m, LocalIndex first,
                                        LocalIndex last, const Vec3<T1>& center, SphericalMultipole<T3, 4>& Mout)
{
    constexpr int P = 4;

    Mout = 0;
    for (LocalIndex i = first; i < last; i++)
    {
        Vec3<T1> body{x[i], y[i], z[i]};
        Vec3<T1> dX = center - body;

        SphericalMultipole<T3, P> M;
        M[0] = m[i];
        Kernels<0, 0, P - 1>::P2M(M, dX);
        Mout += M;
    }
}

/*! @brief apply gravitational interaction with a multipole to a particle
 */
template<class T1, class T2>
HOST_DEVICE_FUN inline util::tuple<T1, T1, T1, T1>
multipole2Particle(T1 tx, T1 ty, T1 tz, const Vec3<T1>& center, SphericalMultipole<T2, 2>& multipole)
{
    Vec3<T1> body{tx, ty, tz};
    Vec4<T1> acc{0, 0, 0, 0};

    acc = M2P(acc, body, center, multipole, T1(0));
    return {acc[1], acc[2], acc[3], acc[0]};
}

/*! @brief apply gravitational interaction with a multipole to a particle
 */
template<class T1, class T2>
HOST_DEVICE_FUN inline util::tuple<T1, T1, T1, T1>
multipole2Particle(T1 tx, T1 ty, T1 tz, const Vec3<T1>& center, SphericalMultipole<T2, 4>& multipole)
{
    Vec3<T1> body{tx, ty, tz};
    Vec4<T1> acc{0, 0, 0, 0};

    acc = M2P(acc, body, center, multipole, T1(0));
    return {acc[1], acc[2], acc[3], acc[0]};
}

/*! @brief Combine multipoles into a single multipole
 */
template<class T, class MType, std::enable_if_t<IsSpherical<MType>{}, int> = 0>
void multipole2Multipole(int begin, int end, const Vec4<T>& Xout, const Vec4<T>* Xsrc, const MType* Msrc, MType& Mout)
{
    Mout = 0;
    M2M(begin, end, Xout, Xsrc, Msrc, Mout);
}

} // namespace ryoanji
