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
 * @brief Barnes-Hut tree walk to compute gravity forces on particles in leaf cells
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/traversal/traversal.hpp"
#include "cstone/traversal/macs.hpp"
#include "cstone/focus/source_center.hpp"
#include "cartesian_qpole.hpp"
#include <algorithm>


namespace ryoanji
{

struct EwaldData
{
    double ix;
    double iy;
    double iz;
    double hx;
    double hy;
    double hz;
    double hfac_cos;
    double hfac_sin;
};

enum EWALD_GRAVITY_SWITCH
{
    GRAV_ALL                = (1 << 0),
    GRAV_NO_CENTRAL_BOX     = (1 << 1),
    GRAV_NO_REPLICAS        = (1 << 2),
    GRAV_NO_EWALD           = (1 << 3),
    GRAV_NO_EWALD_REALSPACE = (1 << 4),
    GRAV_NO_EWALD_KSPACE    = (1 << 5),
};

/*! @brief apply gravitational interaction with a multipole to a particle
 *
 * @tparam        T1         float or double
 * @tparam        T2         float or double
 * @param[in]     tx         target particle x coordinate
 * @param[in]     ty         target particle y coordinate
 * @param[in]     tz         target particle z coordinate
 * @param[in]     center     source center of mass
 * @param[in]     multipole  multipole source
 * @param[inout]  ugrav      location to add gravitational potential to
 * @return                   tuple(ax, ay, az, u)
 *
 * Note: contribution is added to output
 *
 * Direct implementation of the formulae in Hernquist, 1987 (complete reference in file docstring):
 *
 * monopole:   -M/r^3 * vec(r)
 * quadrupole: Q*vec(r) / r^5 - 5/2 * vec(r)*Q*vec(r) * vec(r) / r^7
 */
template<class T1, class T2>
T2 ewaldEvalMultipole(T1 rx, T1 ry, T1 rz, T1 *gamma, const CartesianQuadrupole<T2>& multipole)
{
    T2 qxx = multipole[Cqi::qxx] + multipole[Cqi::traceQ];
    T2 qyy = multipole[Cqi::qyy] + multipole[Cqi::traceQ];
    T2 qzz = multipole[Cqi::qzz] + multipole[Cqi::traceQ];

    T2 Qrx = rx *                qxx  + ry * multipole[Cqi::qxy] + rz * multipole[Cqi::qxz];
    T2 Qry = rx * multipole[Cqi::qxy] + ry *                qyy  + rz * multipole[Cqi::qyz];
    T2 Qrz = rx * multipole[Cqi::qxz] + ry * multipole[Cqi::qyz] + rz *                qzz;

    T2 rQr = T2(0.5) * (rx * Qrx + ry * Qry + rz * Qrz);

    T2 Qtr = 0.5*(qxx + qyy + qzz);

    rQr /= T2(3.0);
    Qtr /= T2(3.0);

    auto ugrav = T2(0.0);
    ugrav -= gamma[2] * rQr;
    ugrav -= gamma[1] * Qtr;
    ugrav -= gamma[0] * multipole[Cqi::mass];

    return ugrav;
}

template<class T1, class T2>
util::tuple<T1, T1, T1, T1> 
ewaldEvalMultipoleComplete(T1 rx, T1 ry, T1 rz, T1 *gamma, const CartesianQuadrupole<T2>& multipole)
{
    T2 qxx = multipole[Cqi::qxx] + multipole[Cqi::traceQ];
    T2 qyy = multipole[Cqi::qyy] + multipole[Cqi::traceQ];
    T2 qzz = multipole[Cqi::qzz] + multipole[Cqi::traceQ];

    T2 Qrx = rx *                qxx  + ry * multipole[Cqi::qxy] + rz * multipole[Cqi::qxz];
    T2 Qry = rx * multipole[Cqi::qxy] + ry *                qyy  + rz * multipole[Cqi::qyz];
    T2 Qrz = rx * multipole[Cqi::qxz] + ry * multipole[Cqi::qyz] + rz *                qzz;

    Qrx /= T2(3.0);
    Qry /= T2(3.0);
    Qrz /= T2(3.0);

    T2 rQr = T2(0.5) * (rx * Qrx + ry * Qry + rz * Qrz);
    T2 Qtr = T2(0.5) * (qxx + qyy + qzz);

    Qtr /= T2(3.0);

    auto ugrav = 0.0
               - gamma[2] * rQr 
               - gamma[1] * Qtr 
               - gamma[0] * multipole[Cqi::mass];

    auto ax = T1(0.0);
    auto ay = T1(0.0);
    auto az = T1(0.0);
    ax += gamma[2] * Qrx;
    ay += gamma[2] * Qry;
    az += gamma[2] * Qrz;

    T2 Qa = gamma[3] * rQr 
          - gamma[2] * Qtr 
          + gamma[1] * multipole[Cqi::mass];

    ax -= rx * Qa;
    ay -= ry * Qa;
    az -= rz * Qa;

//  T2 Qdx = dx * multipole[Cqi::qxx] + dy * multipole[Cqi::qxy] + dz * multipole[Cqi::qxz];
//  T2 Qdy = dx * multipole[Cqi::qxy] + dy * multipole[Cqi::qyy] + dz * multipole[Cqi::qyz];
//  T2 Qdz = dx * multipole[Cqi::qxz] + dy * multipole[Cqi::qyz] + dz * multipole[Cqi::qzz];

//  T2 rQr = T2(0.5) * (dx * Qdx + dy * Qdy + dz * Qdz);

//  T2 Qa = T2(0.0);
//  Qa += gamma[3] * rQr;
//  Qa += gamma[1] * multipole[Cqi::mass];

//  auto ax = T1(0.0);
//  auto ay = T1(0.0);
//  auto az = T1(0.0);
//  ax += gamma[2] * Qdx;
//  ay += gamma[2] * Qdy;
//  az += gamma[2] * Qdz;

//  ax -= dx * Qa;
//  ay -= dy * Qa;
//  az -= dz * Qa;

    return {ax,ay,az,ugrav};
}

template<class T2>
std::vector<struct EwaldData> ewaldInit(const CartesianQuadrupole<T2>& Mroot, double L, double hCut, double alpha_scale)
{
    std::vector<struct EwaldData> ed_list;

    auto hReps = std::ceil(hCut);
    auto alpha = alpha_scale/L;
    auto k4    = M_PI*M_PI/(alpha*alpha*L*L);
    auto hCut2 = hCut * hCut;

    for (auto hx = -hReps; hx <= hReps; hx++)
    for (auto hy = -hReps; hy <= hReps; hy++)
    for (auto hz = -hReps; hz <= hReps; hz++)
    {
        auto h2 = hx*hx + hy*hy + hz*hz;
        if (h2 == 0 || h2 > hCut2) continue;

        auto g0 = exp(-k4*h2)/(M_PI*h2*L);
        auto g1 =  2*M_PI/L * g0;
        auto g2 = -2*M_PI/L * g1;
        auto g3 =  2*M_PI/L * g2;
        auto g4 = -2*M_PI/L * g3;
        auto g5 =  2*M_PI/L * g4;

        decltype(g0)  gamma1[6] = {g0, 0.0, g2, 0.0, g4, 0.0};
        auto mfac_cos = ewaldEvalMultipole(hx,hy,hz, gamma1, Mroot);

        // This is going to evaluate to zero if we only use quadrapole order
        decltype(g0) gamma2[6] = {0.0, g1, 0.0, g3, 0.0, g5};
        auto mfac_sin = ewaldEvalMultipole(hx,hy,hz, gamma2, Mroot);

        struct EwaldData ed = 
        {
            .ix       = hx,
            .iy       = hy,
            .iz       = hz,
            .hx       = 2*M_PI/L*hx,
            .hy       = 2*M_PI/L*hy,
            .hz       = 2*M_PI/L*hz,
            .hfac_cos = mfac_cos,
            .hfac_sin = mfac_sin,
        };

        ed_list.push_back(ed);
    }

    return ed_list;
}

template<class T1, class T2>
void computeGravityGroupEwald(
        const CartesianQuadrupole<T2>& Mroot, const cstone::SourceCenterType<T1>& Mroot_center, double L, 
        std::vector<struct EwaldData> &ed_list,
        double ewaldCut, int numShells, double alpha_scale,
        const T1* x, const T1* y, const T1* z,
        T1* ax, T1* ay, T1* az, T1* ugrav,
        LocalIndex numParticles,
        int gravity_switch
        )
{
    auto ewaldCut2      = ewaldCut*ewaldCut*L*L;
    auto numEwaldShells = std::max(int(ceil(ewaldCut)), numShells);
    auto alpha          = alpha_scale/L;
    auto alpha2         = alpha*alpha;
    auto k1             = M_PI/(alpha2*L*L*L);
    auto ka             = 2.0*alpha/sqrt(M_PI);

    if (numEwaldShells == 0) return;

    for (auto j = LocalIndex(0); j < numParticles; j++) 
    {
        auto dx      = x[j] - Mroot_center[0];
        auto dy      = y[j] - Mroot_center[1];
        auto dz      = z[j] - Mroot_center[2];

        auto ax_t    = 0.0;
        auto ay_t    = 0.0;
        auto az_t    = 0.0;
        auto ugrav_t = k1 * Mroot[Cqi::mass];

        if (gravity_switch & GRAV_NO_EWALD_REALSPACE) goto ewald_shells_end;

        //printf("numShells %i  numEwaldShells %i  ewaldCut2 %f\n", numShells, numEwaldShells, ewaldCut2);

ewald_shells_begin:

        for (auto ix = -numEwaldShells; ix <= numEwaldShells; ix++)
        for (auto iy = -numEwaldShells; iy <= numEwaldShells; iy++)
        for (auto iz = -numEwaldShells; iz <= numEwaldShells; iz++)
        {
            double gamma[6];

            auto in_hole = 
                   (-numShells <= ix && ix <= numShells) 
                && (-numShells <= iy && iy <= numShells) 
                && (-numShells <= iz && iz <= numShells);

            auto rx = dx - ix*L;
            auto ry = dy - iy*L;
            auto rz = dz - iz*L;
            auto r2  = rx*rx + ry*ry + rz*rz;

            //printf("%2f %2f %2f\n", ix, iy, iz);
            //printf("%2i %2i %2i r2: %23.15f  ewaldCut2: %23.15f  in_hole: %i\n", ix, iy, iz, r2, ewaldCut2, in_hole);
            
            // in other words: !(r2 <= ewaldCut2 || in_hole)
            if (r2 > ewaldCut2 && !in_hole) continue;
            if (ewaldCut2 == 0.0) continue;

            //bool use_small_r_approximation = r2 < 3.0e-3*L*L;       // Gasoline
            bool use_small_r_approximation = r2 < 1.2e-3*L*L;       // PKDGrav3
            //use_small_r_approximation = false;

            if (use_small_r_approximation)
            {
                //
                // For small r, series expand about the origin to avoid errors
                // caused by cancellation of large terms.
                // 
                auto alphan  = ka;
                auto r2a2    = alpha2;

                gamma[0]  = alphan*(r2a2/3.0  - 1.0     );  alphan *= 2*alpha2;
                gamma[1]  = alphan*(r2a2/5.0  - 1.0/3.0 );  alphan *= 2*alpha2;
                gamma[2]  = alphan*(r2a2/7.0  - 1.0/5.0 );  alphan *= 2*alpha2;
                gamma[3]  = alphan*(r2a2/9.0  - 1.0/7.0 );  alphan *= 2*alpha2;
                gamma[4]  = alphan*(r2a2/11.0 - 1.0/9.0 );  alphan *= 2*alpha2;
                gamma[5]  = alphan*(r2a2/13.0 - 1.0/11.0);
            }
            else 
            {
                auto r      = sqrt(r2);
                auto dir    = 1/r;
                auto dir2   = dir*dir;
                auto a      = exp(-r2*alpha2) * ka * dir2;
                auto alphan = T1(1.0);

                gamma[0] = dir * (in_hole ? -erf(alpha*r) : erfc(alpha*r));
                gamma[1] =   gamma[0]*dir2 + a;         alphan *= 2*alpha2;
                gamma[2] = 3*gamma[1]*dir2 + alphan*a;  alphan *= 2*alpha2;
                gamma[3] = 5*gamma[2]*dir2 + alphan*a;  alphan *= 2*alpha2;
                gamma[4] = 7*gamma[3]*dir2 + alphan*a;  alphan *= 2*alpha2;
                gamma[5] = 9*gamma[4]*dir2 + alphan*a;

                //printf("            ] %23.15e %23.15e %23.15e %23.15e %23.15e %23.15e\n", r, dir, a, gamma[0], gamma[1], gamma[2]);
            }

            auto [ax_, ay_, az_, u_] = ewaldEvalMultipoleComplete(rx,ry,rz, gamma, Mroot);
            ax_t    += ax_;
            ay_t    += ay_;
            az_t    += az_;
            ugrav_t -= u_;

            //printf("%2i %2i %2i ] ugrav_t: %23.15f  u_: %23.15f\n", ix, iy, iz, ugrav_t, u_);

            //printf("%2i %2i %2i ] ugrav_t: %23.15f  u_: %23.15f  a: %23.15e %23.15e %23.15e\n", ix, iy, iz, ugrav_t, u_, ax_, ay_, az_);
        }

ewald_shells_end:

ewald_waves_begin:

        if (gravity_switch & GRAV_NO_EWALD_KSPACE) goto ewald_waves_end;

        for (auto ed : ed_list)
        {
            auto hdotx = ed.hx*dx + ed.hy*dy + ed.hz*dz;
            auto c = std::cos(hdotx);
            auto s = std::sin(hdotx);
            auto cs_diff = ed.hfac_cos*s - ed.hfac_sin*c;
            auto cs_sum  = ed.hfac_cos*s + ed.hfac_sin*c;

            ax_t    += ed.hx * cs_diff;
            ay_t    += ed.hy * cs_diff;
            az_t    += ed.hz * cs_diff;
            ugrav_t -=         cs_sum;
        }
ewald_waves_end:

        ax[j]    += ax_t;
        ay[j]    += ay_t;
        az[j]    += az_t;
        ugrav[j] += ugrav_t;
    }
}

/*! @brief computes gravitational acceleration for all particles in the specified group
 *
 * @tparam T1                   float or double
 * @tparam T2                   float or double
 * @param[in]    groupIdx       leaf cell index in [0:octree.numLeafNodes()] to compute accelerations for
 * @param[in]    childOffsets   child node index of each node
 * @param[in]    internalToLeaf map to convert an octree node index into a cstone leaf index
 * @param[in]    multipoles     array of length @p octree.numTreeNodes() with the multipole moments for all nodes
 * @param[in]    layout         array of length @p octree.numLeafNodes()+1, layout[i] is the start offset
 *                              into the x,y,z,m arrays for the leaf node with index i. The last element
 *                              is equal to the length of the x,y,z,m arrays.
 * @param[in]    x              x-coordinates
 * @param[in]    y              y-coordinates
 * @param[in]    z              z-coordinates
 * @param[in]    h              smoothing lengths
 * @param[in]    m              masses
 * @param[in]    box            global coordinate bounding box
 * @param[in]    theta          accuracy parameter
 * @param[in]    G              gravitational constant
 * @param[inout] ax             location to add x-acceleration to
 * @param[inout] ay             location to add y-acceleration to
 * @param[inout] az             location to add z-acceleration to
 * @param[inout] ugrav          location to add gravitational potential to
 *
 * Note: acceleration output is added to destination
 */
template<class MType, class T1, class T2, class Tm>
void computeGravityGroupReplica(TreeNodeIndex groupIdx, const TreeNodeIndex* childOffsets, const TreeNodeIndex* internalToLeaf,
                                const cstone::SourceCenterType<T1>* centers, MType* multipoles, const LocalIndex* layout,
                                const T1 dx_target, const T1 dy_target, const T1 dz_target,
                                const T1* x, const T1* y, const T1* z, const T2* h, const Tm* m, float G, T1* ax, T1* ay,
                                T1* az, T1* ugrav)
{
    LocalIndex firstTarget = layout[groupIdx];
    LocalIndex lastTarget  = layout[groupIdx + 1];
    LocalIndex numTargets  = lastTarget - firstTarget;

    Vec3<T1> tMin{x[0], y[0], z[0]};
    Vec3<T1> tMax = tMin;
    for (LocalIndex i = 0; i < numTargets; ++i)
    {
        Vec3<T1> tp{x[i], y[i], z[i]};
        tMin = min(tp, tMin);
        tMax = max(tp, tMax);
    }

    Vec3<T1> targetCenter = (tMax + tMin) * T2(0.5);
    Vec3<T1> targetSize   = (tMax - tMin) * T2(0.5);

    targetCenter += Vec3<T1> {dx_target, dy_target, dz_target};

    /*! @brief octree traversal continuation criterion
     *
     * This functor takes an octree node index as argument. It then evaluates the MAC between
     * the targetBox and the argument octree node. Returns true if the MAC failed to signal
     * the traversal routine to keep going. If the MAC passed, the multipole moments are applied
     * to the particles in the target box and traversal is stopped.
     */
    auto descendOrM2P = 
        [
            firstTarget, lastTarget, centers, multipoles, 
            x, y, z, G, ax, ay, az, ugrav, &targetCenter, &targetSize,
            dx_target, dy_target, dz_target
        ](TreeNodeIndex idx)
    {
        const auto& com = centers[idx];
        const auto& p   = multipoles[idx];

        bool violatesMac = cstone::evaluateMac(makeVec3(com), com[3], targetCenter, targetSize);
        //violatesMac = false;

        if (!violatesMac)
        {
            LocalIndex numTargets = lastTarget - firstTarget;

// apply multipole to all particles in group
#if defined(__llvm__) || defined(__clang__)
#pragma clang loop vectorize(enable)
#endif
            for (LocalIndex t = 0; t < numTargets; ++t)
            {
                LocalIndex offset        = t + firstTarget;
                auto x_target = x[offset] + dx_target;
                auto y_target = y[offset] + dy_target;
                auto z_target = z[offset] + dz_target;

                auto [ax_, ay_, az_, u_] = multipole2Particle(x_target, y_target, z_target, makeVec3(com), p);
                *(ax + t) += G * ax_;
                *(ay + t) += G * ay_;
                *(az + t) += G * az_;
                *(ugrav + t) += G * u_;
            }
        }

        return violatesMac;
    };

    /*! @brief traversal endpoint action
     *
     * This functor gets called with an octree leaf node index whenever traversal hits a leaf node
     * and the leaf failed the MAC w.r.t to the target box. In that case, direct particle-particle
     * interactions need to be computed.
     */
    auto leafP2P = 
        [
            groupIdx, internalToLeaf, layout, firstTarget, lastTarget, 
            x, y, z, h, m, G, ax, ay, az, ugrav,
            dx_target, dy_target, dz_target
        ](TreeNodeIndex idx)
    {
        TreeNodeIndex lidx       = internalToLeaf[idx];
        LocalIndex    numTargets = lastTarget - firstTarget;

        LocalIndex firstSource = layout[lidx];
        LocalIndex lastSource  = layout[lidx + 1];
        LocalIndex numSources  = lastSource - firstSource;

        // source node != target node
        for (LocalIndex t = 0; t < numTargets; ++t)
        {
            LocalIndex offset = t + firstTarget;
            auto x_target = x[offset] + dx_target;
            auto y_target = y[offset] + dy_target;
            auto z_target = z[offset] + dz_target;
            auto [ax_, ay_, az_, u_] =
                particle2Particle(x_target, y_target, z_target, h[offset], 
                                  x + firstSource, y + firstSource, z + firstSource, h + firstSource, m + firstSource, 
                                  numSources);
            *(ax + t) += G * ax_;
            *(ay + t) += G * ay_;
            *(az + t) += G * az_;
            *(ugrav + t) += G * u_;

            //if (t == 0)
            //printf("leaf %3i] %3i %3i %3i  %23.15f  %23.15f  %23.15f, %23.15f\n", t, int(dx_target)/2, int(dy_target)/2, int(dz_target)/2, ax_, ay_, az_, u_);
            //printf("leaf %3i] %23.15f  %23.15f  %23.15f, %23.15f  %23.15f  %23.15f, %23.15f\n", t, x[offset], y[offset], z[offset], ax_, ay_, az_, u_);
            //printf("leaf %3i] %23.15f  %23.15f  %23.15f, %23.15f  %23.15f  %23.15f, %23.15f\n", t, x_target, y_target, z_target, ax_, ay_, az_, u_);
            //printf("leaf %3i] %23.15f  %23.15f  %23.15f, %23.15f  %23.15f  %23.15f, %23.15f\n", t, x_target, y_target, z_target, x[firstSource], y[firstSource], z[firstSource], u_);
        }
    };

    cstone::singleTraversal(childOffsets, descendOrM2P, leafP2P);
}

/*! @brief computes a gravitational correcttion using Ewald summation
 *
 * only computes total gravitational energy, no potential per particle
 *
 * @tparam KeyType               unsigned 32- or 64-bit integer type
 * @tparam T1                    float or double
 * @tparam T2                    float or double
 * @param[in]    childOffsets    child node index of each node
 * @param[in]    internalToLeaf  map to convert an octree node index into a cstone leaf index
 * @param[in]    numLeafNodes    number of leaf nodes
 * @param[in]    multipoles      array of length @p octree.numTreeNodes() with the multipole moments for all nodes
 * @param[in]    layout          array of length @p octree.numLeafNodes()+1, layout[i] is the start offset
 *                               into the x,y,z,m arrays for the leaf node with index i. The last element
 *                               is equal to the length of the x,y,z,m arrays.
 * @param[in]    firstLeafIndex
 * @param[in]    lastLeafIndex
 * @param[in]    x               x-coordinates
 * @param[in]    y               y-coordinates
 * @param[in]    z               z-coordinates
 * @param[in]    h               smoothing lengths
 * @param[in]    m               masses
 * @param[in]    G               gravitational constant
 * @param[inout] ax              location to add x-acceleration to
 * @param[inout] ay              location to add y-acceleration to
 * @param[inout] az              location to add z-acceleration to
 * @return                       total gravitational energy
 */
template<class MType, class T1, class T2, class Tm>
T2 computeGravityEwald(const TreeNodeIndex* childOffsets, const TreeNodeIndex* internalToLeaf, TreeNodeIndex numLeafNodes,
                     const cstone::SourceCenterType<T1>* centers, MType* multipoles, const LocalIndex* layout,
                     TreeNodeIndex firstLeafIndex, TreeNodeIndex lastLeafIndex, const T1* x, const T1* y, const T1* z,
                     const T2* h, const Tm* m, float G, T1* ax, T1* ay, T1* az,
                     const cstone::Box<T1>& box, int numShells = 1, float hCut = 2.8, float ewaldCut = 2.6, float alpha_scale = 2.0, int gravity_switch = GRAV_ALL)
{
    T1 egravTot = 0.0;

    // determine maximum leaf particle count, bucketSize does not work, since octree might not be converged
    std::size_t maxNodeCount = 0;
#pragma omp parallel for reduction(max : maxNodeCount)
    for (TreeNodeIndex i = 0; i < numLeafNodes; ++i)
    {
        maxNodeCount = std::max(maxNodeCount, std::size_t(layout[i + 1] - layout[i]));
    }

    auto L = box.lx();

    std::vector<struct EwaldData> ed_list;

    if (!(gravity_switch & GRAV_NO_EWALD))
    {
       ed_list = ewaldInit(multipoles[0], L, hCut, alpha_scale);
    }


#pragma omp parallel
    {
        T1 ugravThread[maxNodeCount];
        T1 egravThread = 0.0;

#pragma omp for
        for (TreeNodeIndex leafIdx = firstLeafIndex; leafIdx < lastLeafIndex; ++leafIdx)
        {
            std::fill(ugravThread, ugravThread + maxNodeCount, 0);

            LocalIndex firstTarget = layout[leafIdx];
            LocalIndex lastTarget  = layout[leafIdx + 1];
            LocalIndex numTargets  = lastTarget - firstTarget;

            for (auto iz = -numShells; iz <= numShells; iz++)
            for (auto iy = -numShells; iy <= numShells; iy++)
            for (auto ix = -numShells; ix <= numShells; ix++)
            {
                if (ix == 0 && iy == 0 && iz == 0)
                {
                    if (gravity_switch & GRAV_NO_CENTRAL_BOX) continue;

                    computeGravityGroup(leafIdx, childOffsets, internalToLeaf, centers, multipoles, layout, 
                                        x, y, z, h, m, G,
                                        ax + firstTarget, ay + firstTarget, az + firstTarget, ugravThread);
                }
                else
                {
                    auto dx = -ix * L;
                    auto dy = -iy * L;
                    auto dz = -iz * L;

                    if (gravity_switch & GRAV_NO_REPLICAS) continue;

                    computeGravityGroupReplica(leafIdx, childOffsets, internalToLeaf, centers, multipoles, layout, 
                                               dx, dy, dz,
                                               x, y, z, h, m, G,
                                               ax + firstTarget, ay + firstTarget, az + firstTarget, ugravThread);
                }
            }

            if (!(gravity_switch & GRAV_NO_EWALD))
            {
                computeGravityGroupEwald(
                    multipoles[0], centers[0], L, ed_list, ewaldCut, numShells, alpha_scale,
                     x + firstTarget,
                     y + firstTarget,
                     z + firstTarget, 
                    ax + firstTarget,
                    ay + firstTarget,
                    az + firstTarget,
                    ugravThread,
                    numTargets,
                    gravity_switch);
            }

            for (LocalIndex i = 0; i < numTargets; ++i)
            {
                egravThread += m[i + firstTarget] * ugravThread[i];
            }
        }

#pragma omp atomic
        egravTot += egravThread;
    }

    return 0.5 * egravTot;
}

} // namespace ryoanji
