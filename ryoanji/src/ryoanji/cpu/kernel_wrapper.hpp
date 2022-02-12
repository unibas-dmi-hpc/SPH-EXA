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

namespace cstone
{

/*! @brief Compute the monopole and quadruple moments from particle coordinates
 */
template<class T1, class T2, class T3, size_t P>
HOST_DEVICE_FUN void particle2Multipole(const T1* x,
                        const T1* y,
                        const T1* z,
                        const T2* m,
                        LocalIndex first,
                        LocalIndex last,
                        Vec3<T1> center,
                        ryoanji::SphericalMultipole<T3, P>& M)
{
    setZero(M);
}

/*! @brief compute a single particle-particle gravitational interaction
 */
//template<class T1, class T2>
//HOST_DEVICE_FUN inline __attribute__((always_inline)) util::tuple<T1, T1, T1, T1>
//particle2particle(T1 tx, T1 ty, T1 tz, T2 th, T1 sx, T1 sy, T1 sz, T2 sh, T2 sm)
//{
//}

/*! @brief apply gravitational interaction with a multipole to a particle
 */
//template<class T1, class T2>
//HOST_DEVICE_FUN inline util::tuple<T1, T1, T1, T1>
//multipole2particle(T1 tx, T1 ty, T1 tz, const Vec3<T1>& center, const CartesianQuadrupole<T2>& multipole)
//{
//}

/*! @brief Combine multipoles into a single multipole
 */
//template<class T, class MType>
//void multipole2multipole(int begin, int end, const Vec4<T>& Xout, const Vec4<T>* Xsrc, const MType* Msrc, MType& Mout)
//{
//    setZero(Mout);
//}

} // namespace cstone
