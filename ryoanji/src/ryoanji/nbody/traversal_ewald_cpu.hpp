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
 * @brief Barnes-Hut tree walk with Ewald periodic boundaries
 *
 * @author Jonathan Coles <jonathan.coles@cscs.ch>
 */

#pragma once

#include "cstone/traversal/traversal.hpp"
#include "cstone/traversal/macs.hpp"
#include "cstone/focus/source_center.hpp"
#include "cartesian_qpole.hpp"
#include <algorithm>

namespace ryoanji
{

/*! @brief Coefficients used by the sum over k-space.
 *
 */
template<class T1>
struct EwaldHSumCoefficients
{
    Vec3<T1> hr;
    Vec3<T1> hr_scaled;
    T1       hfac_cos;
    T1       hfac_sin;
};

/*! @brief Parameters used by the real- and k-space routines.
 *
 */
template<class T1, class T2>
struct EwaldParameters
{
    CartesianQuadrupole<T2>                Mroot;
    Vec3<T1>                               Mroot_center;
    int                                    numReplicaShells;
    int                                    numEwaldShells;
    double                                 L;
    double                                 lCut;
    double                                 hCut;
    double                                 alpha_scale;
    double                                 small_R_scale_factor;
    std::vector<EwaldHSumCoefficients<T1>> hsum_coeffs;
};

template<class T>
using CartesianQuadrupoleGamma = util::array<T, 6>;

/*! @brief Evaluate the potential and acceleration of a multipole expansion.
 *
 * @tparam        T1         float or double
 * @tparam        T2         float or double
 * @param[in]     rx         target x coordinate relative to multipole center
 * @param[in]     ry         target y coordinate relative to multipole center
 * @param[in]     rz         target z coordinate relative to multipole center
 * @param[in]     gamma      multipole evaluation coefficients
 * @param[in]     multipole  multipole source
 * @return                   tupole(u,ax,ay,az)
 *
 */
template<class T1, class T2>
Vec4<T1> ewaldEvalMultipoleComplete(Vec4<T1> potAcc, Vec3<T1> r, const CartesianQuadrupoleGamma<T2>& gamma,
                                    const CartesianQuadrupole<T2>& multipole)
{
    const T2 M   = multipole[Cqi::mass];
    const T2 qxx = (multipole[Cqi::qxx] + multipole[Cqi::trace]) / T2(3.0);
    const T2 qyy = (multipole[Cqi::qyy] + multipole[Cqi::trace]) / T2(3.0);
    const T2 qzz = (multipole[Cqi::qzz] + multipole[Cqi::trace]) / T2(3.0);
    const T2 qxy = (multipole[Cqi::qxy]) / T2(3.0);
    const T2 qxz = (multipole[Cqi::qxz]) / T2(3.0);
    const T2 qyz = (multipole[Cqi::qyz]) / T2(3.0);

    Vec3<T2> Qr{r[0] * qxx + r[1] * qxy + r[2] * qxz, r[0] * qxy + r[1] * qyy + r[2] * qyz,
                r[0] * qxz + r[1] * qyz + r[2] * qzz};

    const T2 rQr = 0.5 * dot(r, Qr);
    const T2 Qtr = 0.5 * multipole[Cqi::trace];

    auto ugrav = -gamma[0] * M + gamma[1] * Qtr - gamma[2] * rQr;

    auto a = +gamma[2] * Qr - r * (+gamma[1] * M - gamma[2] * Qtr + gamma[3] * rQr);

    return potAcc + Vec4<T1>{ugrav, a[0], a[1], a[2]};
}

/*! @brief Compute Ewald coefficients for k-space summation.
 *
 * @tparam        T2                float or double
 * @param[in]     Mroot             top-level box multipole
 * @param[in]     Mroot_center      center of the multipole expansion
 * @param[in]     numReplicaShells  number of replica shells by tree gravity
 * @param[in]     L                 box side-length
 * @param[in]     lCut              cutoff distance for Ewald l sum (real space) (recommended: 2.6)
 * @param[in]     hCut              cutoff distance for Ewald h sum (k-space) (recommended: 2.8)
 * @param[in]     alpha_scale       scale factor for the mixing factor alpha (recommended: 2.0)
 * @return                          the initialized EwaldParameters struct
 *
 * Evaluates the monopole of the given multipole expansion at the position
 * (rx,ry,rz) relative to the multipole center.
 */
template<class T1, class T2>
EwaldParameters<T1, T2> ewaldInitParameters(const CartesianQuadrupole<T2>& Mroot, const Vec3<T1>& Mroot_center,
                                            int numReplicaShells, double L, double lCut, double hCut,
                                            double alpha_scale)
{
    //
    // This will disable Ewald, but still allow a normal gravity calculation using replicas.
    //
    if (lCut == 0 && hCut == 0 && alpha_scale == 0) { numReplicaShells = 0; }

    EwaldParameters<T1, T2> params = {.Mroot            = Mroot,
                                      .Mroot_center     = Mroot_center,
                                      .numReplicaShells = numReplicaShells,
                                      .numEwaldShells   = std::max(int(ceil(lCut)), numReplicaShells),
                                      .L                = L,
                                      .lCut             = lCut,
                                      .hCut             = hCut,
                                      .alpha_scale      = alpha_scale,
                                      //.small_R_scale_factor = 1.2e-3,    // PKDGrav3, ChaNGa
                                      .small_R_scale_factor = 3.0e-3, // Gasoline
                                      .hsum_coeffs          = {}};

    if (params.numEwaldShells == 0) { return params; }

    const auto hReps = int(std::ceil(hCut));
    const auto alpha = alpha_scale / L;
    const auto k4    = M_PI * M_PI / (alpha * alpha * L * L);
    const auto hCut2 = hCut * hCut;

    for (auto hx = -hReps; hx <= hReps; hx++)
    {
        for (auto hy = -hReps; hy <= hReps; hy++)
        {
            for (auto hz = -hReps; hz <= hReps; hz++)
            {
                Vec3<T1> hr = {T1(hx), T1(hy), T1(hz)};

                const auto h2 = norm2(hr);
                if (h2 == 0) continue;
                if (h2 > hCut2) continue;

                const auto g0 = exp(-k4 * h2) / (M_PI * h2 * L);
                const auto g1 = 2 * M_PI / L * g0;
                const auto g2 = -2 * M_PI / L * g1;
                const auto g3 = 2 * M_PI / L * g2;
                const auto g4 = -2 * M_PI / L * g3;
                const auto g5 = 2 * M_PI / L * g4;

                CartesianQuadrupoleGamma<T2> gamma1{g0, 0.0, g2, 0.0, g4, 0.0};
                const auto                   mfac_cos = ewaldEvalMultipoleComplete({0}, hr, gamma1, Mroot)[0];

                CartesianQuadrupoleGamma<T2> gamma2{0.0, g1, 0.0, g3, 0.0, g5};
                const auto                   mfac_sin = ewaldEvalMultipoleComplete({0}, hr, gamma2, Mroot)[0];

                EwaldHSumCoefficients<T1> hsum = {
                    .hr        = hr,
                    .hr_scaled = 2 * M_PI / L * hr,
                    .hfac_cos  = mfac_cos,
                    .hfac_sin  = mfac_sin,
                };

                params.hsum_coeffs.push_back(hsum);
            }
        }
    }

    return params;
}

//! @brief real space Ewald contribution to the potential and acceleration of a single particle @p r
template<class T1, class T2>
Vec4<T1> computeEwaldRealSpace(Vec3<T1> r, const EwaldParameters<T1, T2>& params)
{
    const auto lCut             = params.lCut;
    const auto alpha_scale      = params.alpha_scale;
    const auto L                = params.L;
    const auto Mroot            = params.Mroot;
    const auto numEwaldShells   = params.numEwaldShells;
    const auto numReplicaShells = params.numReplicaShells;

    const auto lCut2    = lCut * lCut * L * L;
    const auto alpha    = alpha_scale / L;
    const auto alpha2   = alpha * alpha;
    const auto k1       = M_PI / (alpha2 * L * L * L);
    const auto ka       = 2.0 * alpha / sqrt(M_PI);
    const auto small_R2 = params.small_R_scale_factor * L * L;

    Vec4<T1> potAcc{k1 * Mroot[Cqi::mass], 0, 0, 0};

    r -= params.Mroot_center;

    for (auto ix = -numEwaldShells; ix <= numEwaldShells; ix++)
    {
        for (auto iy = -numEwaldShells; iy <= numEwaldShells; iy++)
        {
            for (auto iz = -numEwaldShells; iz <= numEwaldShells; iz++)
            {
                CartesianQuadrupoleGamma<T2> gamma;

                //
                // This is the region that was computed with the normal gravity routine.
                // We need to know this below to add a correction.
                //
                const auto in_precomputed_region = (-numReplicaShells <= ix && ix <= numReplicaShells) &&
                                                   (-numReplicaShells <= iy && iy <= numReplicaShells) &&
                                                   (-numReplicaShells <= iz && iz <= numReplicaShells);

                Vec3<T1> Rrep{ix * L, iy * L, iz * L};

                const auto R  = r + Rrep;
                const auto R2 = norm2(R);

                // in other words: !(r2 <= lCut2 || in_precomputed_region)
                if (R2 > lCut2 && !in_precomputed_region) continue;

                bool use_small_r_approximation = R2 < small_R2 && ka > 0;
                if (use_small_r_approximation)
                {
                    //
                    // For small r, series expand about the origin to avoid errors
                    // caused by cancellation of large terms.
                    //
                    auto c0   = ka;
                    auto R2a2 = R2 * alpha2;

                    gamma[0] = c0 * (R2a2 / 3.0 - 1.0);
                    c0 *= 2 * alpha2;
                    gamma[1] = c0 * (R2a2 / 5.0 - 1.0 / 3.0);
                    c0 *= 2 * alpha2;
                    gamma[2] = c0 * (R2a2 / 7.0 - 1.0 / 5.0);
                    c0 *= 2 * alpha2;
                    gamma[3] = c0 * (R2a2 / 9.0 - 1.0 / 7.0);
                    c0 *= 2 * alpha2;
                    gamma[4] = c0 * (R2a2 / 11.0 - 1.0 / 9.0);
                    c0 *= 2 * alpha2;
                    gamma[5] = c0 * (R2a2 / 13.0 - 1.0 / 11.0);
                }
                else
                {
                    const auto Rmag  = sqrt(R2);
                    const auto invR  = 1.0 / Rmag;
                    const auto invR2 = invR * invR;
                    const auto a     = exp(-R2 * alpha2) * ka * invR2;

                    auto alphan = T1(1.0);

                    //
                    // If we are in the region already computed by the normal gravity routine
                    // then only add a -erf correction, otherwise we need the full erfc value.
                    //
                    auto fn = in_precomputed_region ? -erf(alpha * Rmag) : erfc(alpha * Rmag);

                    gamma[0] = fn * invR;
                    gamma[1] = gamma[0] * invR2 + a;
                    alphan *= 2 * alpha2;
                    gamma[2] = 3 * gamma[1] * invR2 + alphan * a;
                    alphan *= 2 * alpha2;
                    gamma[3] = 5 * gamma[2] * invR2 + alphan * a;
                    alphan *= 2 * alpha2;
                    gamma[4] = 7 * gamma[3] * invR2 + alphan * a;
                    alphan *= 2 * alpha2;
                    gamma[5] = 9 * gamma[4] * invR2 + alphan * a;
                }

                potAcc = ewaldEvalMultipoleComplete(potAcc, R, gamma, Mroot);
            }
        }
    }

    return potAcc;
}

//! @brief Ewald K-space contribution to the potential and acceleration of a single particle @p r
template<class T1, class T2>
Vec4<T1> computeEwaldKSpace(Vec3<T1> r, const EwaldParameters<T1, T2>& params)
{
    Vec4<T1> potAcc = Vec4<T1>{0, 0, 0, 0};
    Vec3<T1> dr     = r - params.Mroot_center;

    for (auto hd : params.hsum_coeffs)
    {
        auto hdotx   = dot(hd.hr_scaled, dr);
        auto c       = std::cos(hdotx);
        auto s       = std::sin(hdotx);
        auto cs_sum  = hd.hfac_cos * c + hd.hfac_sin * s;
        auto cs_diff = hd.hfac_cos * s - hd.hfac_sin * c;

        potAcc[0] -= cs_sum;
        potAcc[1] += cs_diff * hd.hr_scaled[0];
        potAcc[2] += cs_diff * hd.hr_scaled[1];
        potAcc[3] += cs_diff * hd.hr_scaled[2];
    }

    return potAcc;
}

/*! @brief computes a gravitational correction using Ewald summation
 *
 * only computes total gravitational energy, no potential per particle
 *
 * @tparam KeyType               unsigned 32- or 64-bit integer type
 * @tparam T1                    float or double
 * @tparam T2                    float or double
 * @param[in]    childOffsets    child node index of each node
 * @param[in]    internalToLeaf  map to convert an octree node index into a cstone leaf index
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
 * @param[inout] ugrav           location to add potential per particle to, can be nullptr
 * @param[inout] ax              location to add x-acceleration to
 * @param[inout] ay              location to add y-acceleration to
 * @param[inout] az              location to add z-acceleration to
 * @param[inout] ugravTot        total gravitational energy, one element
 * @param[in]    numReplicaShells  see @a ewaldInitParameters
 * @param[in]    lCut              see @a ewaldInitParameters
 * @param[in]    hCut              see @a ewaldInitParameters
 * @param[in]    alpha_scale       see @a ewaldInitParameters
 */
template<class MType, class T1, class T2, class Tm>
void computeGravityEwald(const TreeNodeIndex* childOffsets, const TreeNodeIndex* internalToLeaf,
                         const cstone::SourceCenterType<T1>* macSpheres, MType* multipoles, const LocalIndex* layout,
                         TreeNodeIndex firstLeafIndex, TreeNodeIndex lastLeafIndex, const T1* x, const T1* y,
                         const T1* z, const T2* h, const Tm* m, const cstone::Box<T1>& box, float G, T1* ugrav, T1* ax,
                         T1* ay, T1* az, T1* ugravTot, int numReplicaShells = 1, double lCut = 2.6, double hCut = 2.8,
                         double alpha_scale = 2.0, bool only_ewald = false)
{
    LocalIndex firstTarget = layout[firstLeafIndex];
    LocalIndex lastTarget  = layout[lastLeafIndex];

    if (box.minExtent() != box.maxExtent()) { throw std::runtime_error("Ewald gravity requires cubic bounding boxes"); }

    T1 Udirect = 0;
    if (!only_ewald)
    {
        computeGravity(childOffsets, internalToLeaf, macSpheres, multipoles, layout, firstLeafIndex, lastLeafIndex, x,
                       y, z, h, m, box, G, ugrav, ax, ay, az, &Udirect, numReplicaShells);
    }
    *ugravTot += Udirect;

    EwaldParameters<T1, T2> ewaldParams = ewaldInitParameters(multipoles[0], makeVec3(macSpheres[0]), numReplicaShells,
                                                              box.lx(), lCut, hCut, alpha_scale);

    if (ewaldParams.numEwaldShells == 0) { return; }

    T1 Uewald = 0;
#pragma omp parallel for reduction(+ : Uewald)
    for (LocalIndex i = firstTarget; i < lastTarget; i++)
    {
        Vec3<T1> target{x[i], y[i], z[i]};
        Vec4<T1> potAcc{0, 0, 0, 0};

        potAcc += computeEwaldRealSpace(target, ewaldParams);
        potAcc += computeEwaldKSpace(target, ewaldParams);

        if (ugrav) { ugrav[i] += G * potAcc[0] * m[i]; }

        Uewald += G * potAcc[0] * m[i];
        ax[i] += G * potAcc[1];
        ay[i] += G * potAcc[2];
        az[i] += G * potAcc[3];
    }

    *ugravTot += 0.5 * Uewald;
}

} // namespace ryoanji
