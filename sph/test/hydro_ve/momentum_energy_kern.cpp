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
 * @brief SPH density kernel tests
 *
 * @author Ruben Cabezon <ruben.cabezon@unibas.ch>
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <vector>

#include "gtest/gtest.h"

#include "sph/hydro_ve/momentum_energy_kern.hpp"
#include "sph/tables.hpp"
#include "../../../main/src/io/file_utils.hpp"

using namespace sph;

template<class T>
void symmetrizeGradV(util::array<const T*, 9> dV, util::array<T*, 6> sdV, size_t n)
{
    for (size_t i = 0; i < n; ++i)
    {
        sdV[0][i] = dV[0][i];
        sdV[1][i] = dV[1][i] + dV[3][i];
        sdV[2][i] = dV[2][i] + dV[6][i];
        sdV[3][i] = dV[4][i];
        sdV[4][i] = dV[5][i] + dV[7][i];
        sdV[5][i] = dV[8][i];
    }
}

TEST(MomentumEnergy, JLoop)
{
    using T = double;

    T sincIndex = 6.0;
    T K         = compute_3d_k(sincIndex);
    T Atmin     = 0.1;
    T Atmax     = 0.2;
    T ramp      = 1.0 / (Atmax - Atmin);
    T mpart     = 3.781038064465603e26;

    std::array<double, lt::size> wh  = lt::createWharmonicTable<double, lt::size>(sincIndex);
    std::array<double, lt::size> whd = lt::createWharmonicDerivativeTable<double, lt::size>(sincIndex);

    cstone::Box<T> box(-1.e9, 1.e9, cstone::BoundaryType::open);

    size_t   npart          = 99;
    unsigned neighborsCount = npart - 1, i;

    std::vector<cstone::LocalIndex> neighbors(neighborsCount - 1);

    for (i = 0; i < neighborsCount; i++)
    {
        neighbors[i] = i + 1;
    }

    std::vector<T> x(npart);
    std::vector<T> y(npart);
    std::vector<T> z(npart);
    std::vector<T> h(npart);
    std::vector<T> m(npart);
    std::vector<T> gradh(npart);
    std::vector<T> rho0(npart);
    std::vector<T> sumwhrho0(npart);
    std::vector<T> vx(npart);
    std::vector<T> vy(npart);
    std::vector<T> vz(npart);
    std::vector<T> c(npart);
    std::vector<T> p(npart);
    std::vector<T> u(npart);
    std::vector<T> divv(npart);
    std::vector<T> alpha(npart);
    std::vector<T> c11(npart);
    std::vector<T> c12(npart);
    std::vector<T> c13(npart);
    std::vector<T> c22(npart);
    std::vector<T> c23(npart);
    std::vector<T> c33(npart);
    std::vector<T> dvxdx(npart);
    std::vector<T> dvxdy(npart);
    std::vector<T> dvxdz(npart);
    std::vector<T> dvydx(npart);
    std::vector<T> dvydy(npart);
    std::vector<T> dvydz(npart);
    std::vector<T> dvzdx(npart);
    std::vector<T> dvzdy(npart);
    std::vector<T> dvzdz(npart);
    std::vector<T> sumwh(npart);
    std::vector<T> xm(npart);
    std::vector<T> kx(npart);

    std::vector<T*> fields{x.data(),     y.data(),     z.data(),     vx.data(),    vy.data(),    vz.data(),
                           h.data(),     c.data(),     c11.data(),   c12.data(),   c13.data(),   c22.data(),
                           c23.data(),   c33.data(),   p.data(),     gradh.data(), rho0.data(),  sumwhrho0.data(),
                           sumwh.data(), dvxdx.data(), dvxdy.data(), dvxdz.data(), dvydx.data(), dvydy.data(),
                           dvydz.data(), dvzdx.data(), dvzdy.data(), dvzdz.data(), alpha.data(), u.data(),
                           divv.data()};

    sphexa::fileutils::readAscii("example_data.txt", npart, fields);

    std::vector<T> dV11(npart);
    std::vector<T> dV12(npart);
    std::vector<T> dV13(npart);
    std::vector<T> dV22(npart);
    std::vector<T> dV23(npart);
    std::vector<T> dV33(npart);
    symmetrizeGradV<T>({dvxdx.data(), dvxdy.data(), dvxdz.data(), dvydx.data(), dvydy.data(), dvydz.data(),
                        dvzdx.data(), dvzdy.data(), dvzdz.data()},
                       {dV11.data(), dV12.data(), dV13.data(), dV22.data(), dV23.data(), dV33.data()}, npart);

    std::fill(m.begin(), m.end(), mpart);

    for (i = 0; i < neighborsCount + 1; i++)
    {
        xm[i] = mpart / rho0[i];
        kx[i] = K * xm[i] / std::pow(h[i], 3);
    }

    std::vector<T> prho(p.size());
    for (size_t k = 0; k < prho.size(); ++k)
    {
        prho[k] = p[k] / (kx[k] * m[k] * m[k] * gradh[k]);
    }

    // test with AV cleaning
    {
        // fill with invalid initial value to make sure that the kernel overwrites it instead of add to it
        T du         = -1;
        T grad_Px    = -1;
        T grad_Py    = -1;
        T grad_Pz    = -1;
        T maxvsignal = -1;

        // compute gradient for for particle 0
        momentumAndEnergyJLoop<true>(0, K, box, neighbors.data(), neighborsCount, x.data(), y.data(), z.data(),
                                     vx.data(), vy.data(), vz.data(), h.data(), m.data(), prho.data(), c.data(),
                                     c11.data(), c12.data(), c13.data(), c22.data(), c23.data(), c33.data(), Atmin,
                                     Atmax, ramp, wh.data(), whd.data(), kx.data(), xm.data(), alpha.data(),
                                     dV11.data(), dV12.data(), dV13.data(), dV22.data(), dV23.data(), dV33.data(),
                                     &grad_Px, &grad_Py, &grad_Pz, &du, &maxvsignal);

        EXPECT_NEAR(grad_Px, -505548.68073726865, 0.023);
        EXPECT_NEAR(grad_Py, 303384.91384746187, 0.053);
        EXPECT_NEAR(grad_Pz, -1767463.9739728321, 0.043);
        EXPECT_NEAR(du, 8.5525242525359648e12, 7.1e5);
        EXPECT_NEAR(maxvsignal, 26490876.319252387, 1e-6);
    }
    // test without AV cleaning
    {
        T du         = -1;
        T grad_Px    = -1;
        T grad_Py    = -1;
        T grad_Pz    = -1;
        T maxvsignal = -1;

        // compute gradient for for particle 0
        momentumAndEnergyJLoop<false>(0, K, box, neighbors.data(), neighborsCount, x.data(), y.data(), z.data(),
                                      vx.data(), vy.data(), vz.data(), h.data(), m.data(), prho.data(), c.data(),
                                      c11.data(), c12.data(), c13.data(), c22.data(), c23.data(), c33.data(), Atmin,
                                      Atmax, ramp, wh.data(), whd.data(), kx.data(), xm.data(), alpha.data(),
                                      dV11.data(), dV12.data(), dV13.data(), dV22.data(), dV23.data(), dV33.data(),
                                      &grad_Px, &grad_Py, &grad_Pz, &du, &maxvsignal);

        EXPECT_NEAR(grad_Px, -521261.07791667967, 0.022);
        EXPECT_NEAR(grad_Py, -74471.016515749841, 0.064);
        EXPECT_NEAR(grad_Pz, -1730426.827721074, 0.042);
        EXPECT_NEAR(du, 7.1838438980436924e12, 3.1e5);
        EXPECT_NEAR(maxvsignal, 26490876.319252387, 1e-6);
    }
}
