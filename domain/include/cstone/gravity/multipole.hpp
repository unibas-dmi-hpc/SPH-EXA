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
 * @author Mohamed Ayoub Neggaz
 *
 * See for example Hernquist 1987, Performance Characteristics of Tree Codes,
 * https://ui.adsabs.harvard.edu/abs/1987ApJS...64..715H
 *
 */

#pragma once

#include <cmath>

#include "cstone/tree/definitions.h"

namespace cstone
{

template <class T>
struct GravityMultipole
{
    //! @brief geometric center
    T xce = 0.0 , yce = 0.0, zce = 0.0;

    // monopole

    //! @brief total mass
    T mass = 0.0;
    //! @brief center of mass
    T xcm = 0.0, ycm = 0.0, zcm = 0.0;

    // quadrupole

    //! @brief quadrupole moments w.r.t to center of mass
    T qxx = 0.0, qxy = 0.0, qxz = 0.0;
    T qyy = 0.0, qyz = 0.0;
    T qzz = 0.0;

    //! @brief quadrupole moments w.r.t to geometric center
    T qxxa = 0.0, qxya = 0.0, qxza = 0.0;
    T qyya = 0.0, qyza = 0.0;
    T qzza = 0.0;
};

/*! @brief Compute the monopole and quadruple moments from particle coordinates
 *
 * @tparam T              float or double
 * @param  x              x coordinate array
 * @param  y              y coordinate array
 * @param  z              z coordinate array
 * @param  m              Vector of masses.
 * @param  numParticles   number of particles to read from coordinate arrays
 */
template<class T>
GravityMultipole<T> particle2Multipole(const T* x, const T* y, const T* z, const T* m, LocalParticleIndex numParticles,
                                       T xce, T yce, T zce)
{
    GravityMultipole<T> gv;

    gv.xce = xce;
    gv.yce = yce;
    gv.zce = zce;

    for (LocalParticleIndex i = 0; i < numParticles; ++i)
    {
        T xx = x[i];
        T yy = y[i];
        T zz = z[i];

        T m_i = m[i];

        gv.xcm += xx * m_i;
        gv.ycm += yy * m_i;
        gv.zcm += zz * m_i;

        gv.mass += m_i;

        T rx = xx - gv.xce;
        T ry = yy - gv.yce;
        T rz = zz - gv.zce;

        gv.qxxa += rx * rx * m_i;
        gv.qxya += rx * ry * m_i;
        gv.qxza += rx * rz * m_i;
        gv.qyya += ry * ry * m_i;
        gv.qyza += ry * rz * m_i;
        gv.qzza += rz * rz * m_i;
    }

    gv.xcm /= gv.mass;
    gv.ycm /= gv.mass;
    gv.zcm /= gv.mass;

    T rx = gv.xce - gv.xcm;
    T ry = gv.yce - gv.ycm;
    T rz = gv.zce - gv.zcm;
    gv.qxx = gv.qxxa - rx * rx * gv.mass;
    gv.qxy = gv.qxya - rx * ry * gv.mass;
    gv.qxz = gv.qxza - rx * rz * gv.mass;
    gv.qyy = gv.qyya - ry * ry * gv.mass;
    gv.qyz = gv.qyza - ry * rz * gv.mass;
    gv.qzz = gv.qzza - rz * rz * gv.mass;

    return gv;
}

/*! @brief direct gravity calculation with particle-particle interactions
 *
 * @tparam       T             float or double
 * @param[in]    xt            target particle x coordinate
 * @param[in]    yt            target particle y coordinate
 * @param[in]    zt            target particle z coordinate
 * @param[in]    xs            source particle x coordinates
 * @param[in]    ys            source particle y coordinates
 * @param[in]    zs            source particle z coordinates
 * @param[in]    m
 * @param[in]    numSources    number of source particles
 * @param[in]    eps2          square of softening parameter epsilon
 * @param[inout] xacc          location to add x-acceleration to
 * @param[inout] yacc          location to add y-acceleration to
 * @param[inout] zacc          location to add z-acceleration to
 *
 * Computes direct particle-particle gravitational interaction according to
 *
 *      a_t = - sum_{j} m_j / (r_tj^2 + eps2)^(3/2)) * (r_t - r_j)
 *
 * Note: contribution is added to output
 */
template<class T>
void particle2particle(T xt, T yt, T zt, const T* xs, const T* ys, const T* zs, const T* m,
                       LocalParticleIndex numSources, T eps2, T* xacc, T* yacc, T* zacc)
{
    for (LocalParticleIndex j = 0; j < numSources; ++j)
    {
        T xj = xs[j];
        T yj = ys[j];
        T zj = zs[j];

        T dx = xj - xt;
        T dy = yj - yt;
        T dz = zj - zt;

        T denom2 = dx*dx + dy*dy + dz*dz + eps2;
        T invDenom = 1.0 / std::sqrt(denom2);
        T invDenom2 = invDenom * invDenom;

        // prefactor is mj / (r^2 + eps^2)^(3/2)
        T prefactor = m[j] * invDenom * invDenom2;

        *xacc += prefactor * dx;
        *yacc += prefactor * dy;
        *zacc += prefactor * dz;
    }
}

/*! @brief apply gravitational interaction with a multipole to a particle
 *
 * @tparam        T          float or double
 * @param[in]     xt         target particle x coordinate
 * @param[in]     yt         target particle y coordinate
 * @param[in]     zt         target particle z coordinate
 * @param[in]     multipole  multipole source
 * @param[in]     eps2       square of softening parameter epsilon
 * @param[inout]  xacc       location to add x-acceleration to
 * @param[inout]  yacc       location to add y-acceleration to
 * @param[inout]  zacc       location to add z-acceleration to
 *
 * Note: contribution is added to output
 */
template<class T>
void multipole2particle(T xt, T yt, T zt, const GravityMultipole<T>& multipole, T eps2, T* xacc, T* yacc, T* zacc)
{
    // monopole: -M/r^3 * vec(r)

    T r1 = xt - multipole.xcm;
    T r2 = yt - multipole.ycm;
    T r3 = zt - multipole.zcm;

    T r_2      = r1*r1 + r2*r2 + r3*r3 + eps2;
    T r_minus1 = 1.0 / std::sqrt(r_2);
    T r_minus2 = r_minus1 * r_minus1;

    T Mr_minus3 = multipole.mass * r_minus1 * r_minus2;

    *xacc -= Mr_minus3 * r1;
    *yacc -= Mr_minus3 * r2;
    *zacc -= Mr_minus3 * r3;

    // quadrupole: Q*vec(r)/r^5 * vec(r) - 5/2 * vec(r)*Q*vec(r) * vec(r) / r^7

    T r_minus5 = r_minus2 * r_minus2 * r_minus1;
    T r_minus7 = r_minus5 * r_minus2;

    T Qr1 = r1 * multipole.qxx + r2 * multipole.qxy + r3 * multipole.qxz;
    T Qr2 = r1 * multipole.qxy + r2 * multipole.qyy + r3 * multipole.qyz;
    T Qr3 = r1 * multipole.qxz + r2 * multipole.qyz + r3 * multipole.qzz;

    T rQr = r1 * Qr1 + r2 * Qr2 + r3 * Qr3;

    T c1 = -7.5 * r_minus7 * rQr;
    T c2 = 3.0 * r_minus5;
    T c3 = 0.5 * (multipole.qxx + multipole.qyy + multipole.qzz);

    *xacc += c1 * r1 + c2 * (Qr1 + c3 * r1);
    *yacc += c1 * r2 + c2 * (Qr2 + c3 * r2);
    *zacc += c1 * r3 + c2 * (Qr3 + c3 * r3);
}

//template <class I, class T>
//void multipole2multipole(const std::vector<I> &tree, const GravityOctree<I, T> &octree, cstone::TreeNodeIndex i,
//                         GravityTree<T> &gravityLeafData, GravityTree<T> &gravityInternalData, const std::vector<T> &x,
//                         const std::vector<T> &y, const std::vector<T> &z, const std::vector<T> &m, const std::vector<I> &codes,
//                         const cstone::Box<T> &box)
//{
//    // cstone::OctreeNode<I> node = localTree.internalTree()[i];
//
//    GravityData<T> gv;
//
//    pair<T> xrange = octree.x(i, box);
//    pair<T> yrange = octree.y(i, box);
//    pair<T> zrange = octree.z(i, box);
//    gv.xce = (xrange[1] + xrange[0]) / 2.0;
//    gv.yce = (yrange[1] + yrange[0]) / 2.0;
//    gv.zce = (zrange[1] + zrange[0]) / 2.0;
//    gv.dx = abs(xrange[1] - xrange[0]);
//
//    for (int j = 0; j < 8; ++j)
//    {
//        // cstone::TreeNodeIndex child = node.child[j];
//        cstone::TreeNodeIndex child = octree.childDirect(i, j);
//        GravityData<T> current;
//        // if (node.childType[j] == cstone::OctreeNode<I>::ChildType::internal) { current = gravityInternalData[child]; }
//        if (!octree.isLeafChild(i, j)) { current = gravityInternalData[child]; }
//        else
//        {
//            current = gravityLeafData[child];
//        }
//        gv.mTot += current.mTot;
//        gv.xcm += current.xcm * current.mTot;
//        gv.ycm += current.ycm * current.mTot;
//        gv.zcm += current.zcm * current.mTot;
//    }
//    gv.xcm /= gv.mTot;
//    gv.ycm /= gv.mTot;
//    gv.zcm /= gv.mTot;
//
//    size_t n = codes.size();
//
//    for (int j = 0; j < 8; ++j)
//    {
//        // cstone::TreeNodeIndex child = node.child[j];
//        cstone::TreeNodeIndex child = octree.childDirect(i, j);
//        GravityData<T> partialGravity;
//        // if (node.childType[j] == cstone::OctreeNode<I>::ChildType::internal) { partialGravity = gravityInternalData[child]; }
//        if (!octree.isLeafChild(i, j)) { partialGravity = gravityInternalData[child]; }
//        else
//        {
//            partialGravity = gravityLeafData[child];
//        }
//
//        T rx = partialGravity.xcm - gv.xcm;
//        T ry = partialGravity.ycm - gv.ycm;
//        T rz = partialGravity.zcm - gv.zcm;
//
//        gv.qxxa += partialGravity.qxx + rx * rx * partialGravity.mTot;
//        gv.qxya += partialGravity.qxy + rx * ry * partialGravity.mTot;
//        gv.qxza += partialGravity.qxz + rx * rz * partialGravity.mTot;
//        gv.qyya += partialGravity.qyy + ry * ry * partialGravity.mTot;
//        gv.qyza += partialGravity.qyz + ry * rz * partialGravity.mTot;
//        gv.qzza += partialGravity.qzz + rz * rz * partialGravity.mTot;
//
//        gv.pcount += partialGravity.pcount;
//    }
//
//    if (gv.pcount == 1) gv.dx = 0;
//
//    gv.qxx = gv.qxxa;
//    gv.qxy = gv.qxya;
//    gv.qxz = gv.qxza;
//    gv.qyy = gv.qyya;
//    gv.qyz = gv.qyza;
//    gv.qzz = gv.qzza;
//
//    gv.trq = gv.qxx + gv.qyy + gv.qzz;
//    gravityInternalData[i] = gv;
//}

} // namespace cstone
