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
 * @brief implements elementary gravity data structures for octree nodes
 *
 * @author Sebastian Keller        <sebastian.f.keller@gmail.com>
 *
 * See for example Hernquist 1987, Performance Characteristics of Tree Codes,
 * https://ui.adsabs.harvard.edu/abs/1987ApJS...64..715H
 */

#pragma once

#include <cmath>

#include "cstone/tree/definitions.h"
#include "cstone/util/tuple.hpp"

namespace cstone
{

template <class T>
struct GravityMultipole
{
    //! @brief total mass
    T mass = 0.0;

    //! @brief center of mass
    T xcm = 0.0, ycm = 0.0, zcm = 0.0;

    //! @brief quadrupole moments
    T qxx = 0.0, qxy = 0.0, qxz = 0.0;
    T qyy = 0.0, qyz = 0.0;
    T qzz = 0.0;
};

/*! @brief Compute the monopole and quadruple moments from particle coordinates
 *
 * @tparam T1             float or double
 * @tparam T2             float or double
 * @tparam T3             float or double
 * @param  x              x coordinate array
 * @param  y              y coordinate array
 * @param  z              z coordinate array
 * @param  m              Vector of masses.
 * @param  numParticles   number of particles to read from coordinate arrays
 */
template<class T1, class T2, class T3>
GravityMultipole<T1>
particle2Multipole(const T2* x, const T2* y, const T2* z, const T3* m, LocalParticleIndex numParticles)
{
    GravityMultipole<T1> gv;

    if (numParticles == 0) { return gv; }

    // choose position of the first source particle as the expansion center
    T1 xce = x[0];
    T1 yce = y[0];
    T1 zce = z[0];

    for (LocalParticleIndex i = 0; i < numParticles; ++i)
    {
        T1 xx = x[i];
        T1 yy = y[i];
        T1 zz = z[i];

        T1 m_i = m[i];

        gv.xcm += xx * m_i;
        gv.ycm += yy * m_i;
        gv.zcm += zz * m_i;

        gv.mass += m_i;

        T1 rx = xx - xce;
        T1 ry = yy - yce;
        T1 rz = zz - zce;

        gv.qxx += rx * rx * m_i;
        gv.qxy += rx * ry * m_i;
        gv.qxz += rx * rz * m_i;
        gv.qyy += ry * ry * m_i;
        gv.qyz += ry * rz * m_i;
        gv.qzz += rz * rz * m_i;
    }

    gv.xcm /= gv.mass;
    gv.ycm /= gv.mass;
    gv.zcm /= gv.mass;

    T1 rx = xce - gv.xcm;
    T1 ry = yce - gv.ycm;
    T1 rz = zce - gv.zcm;

    // move expansion center to center of mass
    gv.qxx = gv.qxx - rx * rx * gv.mass;
    gv.qxy = gv.qxy - rx * ry * gv.mass;
    gv.qxz = gv.qxz - rx * rz * gv.mass;
    gv.qyy = gv.qyy - ry * ry * gv.mass;
    gv.qyz = gv.qyz - ry * rz * gv.mass;
    gv.qzz = gv.qzz - rz * rz * gv.mass;

    T1 traceQ = gv.qxx + gv.qyy + gv.qzz;

    // remove trace
    gv.qxx = 3 * gv.qxx - traceQ;
    gv.qyy = 3 * gv.qyy - traceQ;
    gv.qzz = 3 * gv.qzz - traceQ;
    gv.qxy *= 3;
    gv.qxz *= 3;
    gv.qyz *= 3;

    return gv;
}

/*! @brief direct gravity calculation with particle-particle interactions
 *
 * @tparam       T1          float or double
 * @tparam       T2          float or double
 * @param[in]    tx          target particle x coordinate
 * @param[in]    ty          target particle y coordinate
 * @param[in]    tz          target particle z coordinate
 * @param[in]    sx          source particle x coordinates
 * @param[in]    sy          source particle y coordinates
 * @param[in]    sz          source particle z coordinates
 * @param[in]    m
 * @param[in]    numSources  number of source particles
 * @param[in]    eps2        square of softening parameter epsilon
 * @param[inout] ax          location to add x-acceleration to
 * @param[inout] ay          location to add y-acceleration to
 * @param[inout] az          location to add z-acceleration to
 * @param[inout] ugrav       location to add gravitational potential to
 *
 * Computes direct particle-particle gravitational interaction according to
 *
 *      a_t = - sum_{j} m_j / (r_tj^2 + eps2)^(3/2)) * (r_t - r_j)
 *
 * Notes:
 *  - Contribution is added to output
 *  - Source particles MUST NOT contain the target. If the source is a cell that contains the target,
 *    the target must be located and this function called twice, with all particles before target and
 *    all particles that follow it.
 */
template<class T1, class T2>
void particle2particle(T1 tx, T1 ty, T1 tz, const T1* sx, const T1* sy, const T1* sz, const T2* m,
                       LocalParticleIndex numSources, T1 eps2, T1* ax, T1* ay, T1* az, T1* ugrav)
{
    for (LocalParticleIndex j = 0; j < numSources; ++j)
    {
        T1 rx = sx[j] - tx;
        T1 ry = sy[j] - ty;
        T1 rz = sz[j] - tz;

        T1 r_2 = rx * rx + ry * ry + rz * rz + eps2;
        T1 r_minus1 = 1.0 / std::sqrt(r_2);
        T1 r_minus2 = r_minus1 * r_minus1;

        T1 Mr_minus3 = m[j] * r_minus1 * r_minus2;

        *ax += Mr_minus3 * rx;
        *ay += Mr_minus3 * ry;
        *az += Mr_minus3 * rz;

        *ugrav -= Mr_minus3 * r_2;
    }
}

//! \brief same as above, but mass softening
template<class T1, class T2>
void particle2particle(T1 tx, T1 ty, T1 tz, T2 hi, const T1* sx, const T1* sy, const T1* sz, const T2* h, const T2* m,
                       LocalParticleIndex numSources, T1* ax, T1* ay, T1* az, T1* ugrav)
{
    for (LocalParticleIndex j = 0; j < numSources; ++j)
    {
        T1 rx = sx[j] - tx;
        T1 ry = sy[j] - ty;
        T1 rz = sz[j] - tz;

        T1 r_2 = rx * rx + ry * ry + rz * rz;
        T1 r = std::sqrt(r_2);
        T1 r_minus1 = 1.0 / r;
        T1 r_minus2 = r_minus1 * r_minus1;

        T1 mEffective = m[j];

        T1 hij = hi + h[j];
        if (r < hij)
        {
            // apply mass softening correction
            T1 vgr = r / hij;
            mEffective *= vgr * vgr * vgr;
        }

        T1 Mr_minus3 = mEffective * r_minus1 * r_minus2;

        *ax += Mr_minus3 * rx;
        *ay += Mr_minus3 * ry;
        *az += Mr_minus3 * rz;

        *ugrav -= Mr_minus3 * r_2;
    }
}

/*! @brief apply gravitational interaction with a multipole to a particle
 *
 * @tparam        T1         float or double
 * @tparam        T2         float or double
 * @param[in]     tx         target particle x coordinate
 * @param[in]     ty         target particle y coordinate
 * @param[in]     tz         target particle z coordinate
 * @param[in]     multipole  multipole source
 * @param[in]     eps2       square of softening parameter epsilon
 * @param[inout]  ax         location to add x-acceleration to
 * @param[inout]  ay         location to add y-acceleration to
 * @param[inout]  az         location to add z-acceleration to
 * @param[inout]  ugrav      location to add gravitational potential to
 *
 * Note: contribution is added to output
 *
 * Direct implementation of the formulae in Hernquist, 1987 (complete reference in file docstring):
 *
 * monopole:   -M/r^3 * vec(r)
 * quadrupole: Q*vec(r) / r^5 - 5/2 * vec(r)*Q*vec(r) * vec(r) / r^7
 */
template<class T1, class T2>
void multipole2particle(T1 tx, T1 ty, T1 tz, const GravityMultipole<T2>& multipole, T1* ax, T1* ay, T1* az, T1* ugrav)
{
    T2 rx = tx - multipole.xcm;
    T2 ry = ty - multipole.ycm;
    T2 rz = tz - multipole.zcm;

    T2 r_2      = rx * rx + ry * ry + rz * rz;
    T2 r_minus1 = 1.0 / std::sqrt(r_2);
    T2 r_minus2 = r_minus1 * r_minus1;
    T2 r_minus5 = r_minus2 * r_minus2 * r_minus1;

    T2 Qrx = rx * multipole.qxx + ry * multipole.qxy + rz * multipole.qxz;
    T2 Qry = rx * multipole.qxy + ry * multipole.qyy + rz * multipole.qyz;
    T2 Qrz = rx * multipole.qxz + ry * multipole.qyz + rz * multipole.qzz;

    T2 rQr = rx * Qrx + ry * Qry + rz * Qrz;
    //                  rQr quad-term           mono-term
    //                      |                     |
    T2 rQrAndMonopole = (-2.5 * rQr * r_minus5 - multipole.mass * r_minus1) * r_minus2;

    //       Qr Quad-term
    *ax += r_minus5 * Qrx + rQrAndMonopole * rx;
    *ay += r_minus5 * Qry + rQrAndMonopole * ry;
    *az += r_minus5 * Qrz + rQrAndMonopole * rz;
    *ugrav -= (multipole.mass * r_minus1 + 0.5 * r_minus5 * rQr);
}

/*! @brief add a multipole contribution to the composite multipole
 *
 * @tparam        T           float or double
 * @param[inout]  composite   the composite multipole
 * @param[in]     addend      the multipole to add
 *
 * Implements formula (2.5) from Hernquist 1987 (parallel axis theorem)
 */
template<class T>
void addQuadrupole(GravityMultipole<T>* composite, const GravityMultipole<T>& addend)
{
    // displacement vector from subcell c-o-m to composite cell c-o-m
    T rx = addend.xcm - composite->xcm;
    T ry = addend.ycm - composite->ycm;
    T rz = addend.zcm - composite->zcm;

    T rx_2 = rx * rx;
    T ry_2 = ry * ry;
    T rz_2 = rz * rz;
    T r_2  = (rx_2 + ry_2 + rz_2) * (1.0/3.0);

    T ml = addend.mass * 3;

    composite->qxx += addend.qxx + ml * (rx_2 - r_2);
    composite->qxy += addend.qxy + ml * rx * ry;
    composite->qxz += addend.qxz + ml * rx * rz;
    composite->qyy += addend.qyy + ml * (ry_2 - r_2);
    composite->qyz += addend.qyz + ml * ry * rz;
    composite->qzz += addend.qzz + ml * (rz_2 - r_2);
}

/*! @brief aggregate multipoles into a composite multipole
 *
 * @tparam T     float or double
 * @param  a..h  subcell multipoles
 * @return       the composite multipole
 *
 * Works for any number of subcell multipoles, but we only need to combine 8 at a time.
 */
template<class T>
GravityMultipole<T> multipole2multipole(const GravityMultipole<T>& a,
                                        const GravityMultipole<T>& b,
                                        const GravityMultipole<T>& c,
                                        const GravityMultipole<T>& d,
                                        const GravityMultipole<T>& e,
                                        const GravityMultipole<T>& f,
                                        const GravityMultipole<T>& g,
                                        const GravityMultipole<T>& h)
{
    GravityMultipole<T> gv;

    gv.mass = a.mass + b.mass + c.mass + d.mass + e.mass + f.mass + g.mass + h.mass;

    gv.xcm  = a.xcm * a.mass + b.xcm * b.mass + c.xcm * c.mass + d.xcm * d.mass;
    gv.ycm  = a.ycm * a.mass + b.ycm * b.mass + c.ycm * c.mass + d.ycm * d.mass;
    gv.zcm  = a.zcm * a.mass + b.zcm * b.mass + c.zcm * c.mass + d.zcm * d.mass;
    gv.xcm += e.xcm * e.mass + f.xcm * f.mass + g.xcm * g.mass + h.xcm * h.mass;
    gv.ycm += e.ycm * e.mass + f.ycm * f.mass + g.ycm * g.mass + h.ycm * h.mass;
    gv.zcm += e.zcm * e.mass + f.zcm * f.mass + g.zcm * g.mass + h.zcm * h.mass;

    gv.xcm /= gv.mass;
    gv.ycm /= gv.mass;
    gv.zcm /= gv.mass;

    addQuadrupole(&gv, a);
    addQuadrupole(&gv, b);
    addQuadrupole(&gv, c);
    addQuadrupole(&gv, d);
    addQuadrupole(&gv, e);
    addQuadrupole(&gv, f);
    addQuadrupole(&gv, g);
    addQuadrupole(&gv, h);

    return gv;
}

} // namespace cstone
