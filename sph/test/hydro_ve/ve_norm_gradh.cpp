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

#include "sph/hydro_ve/ve_def_gradh_kern.hpp"
#include "sph/tables.hpp"
#include "../../../main/src/io/file_utils.hpp"

using namespace sph;

TEST(VeDefGradh, JLoop)
{
    using T = double;

    T sincIndex = 6.0;
    T K         = compute_3d_k(sincIndex);
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
    std::vector<T> gradhp(npart);
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

    std::vector<T*> fields{x.data(),     y.data(),     z.data(),     vx.data(),     vy.data(),    vz.data(),
                           h.data(),     c.data(),     c11.data(),   c12.data(),    c13.data(),   c22.data(),
                           c23.data(),   c33.data(),   p.data(),     gradhp.data(), rho0.data(),  sumwhrho0.data(),
                           sumwh.data(), dvxdx.data(), dvxdy.data(), dvxdz.data(),  dvydx.data(), dvydy.data(),
                           dvydz.data(), dvzdx.data(), dvzdy.data(), dvzdz.data(),  alpha.data(), u.data(),
                           divv.data()};

    sphexa::fileutils::readAscii("example_data.txt", npart, fields);

    std::fill(m.begin(), m.end(), mpart);

    for (i = 0; i < neighborsCount + 1; i++)
    {
        xm[i] = mpart / rho0[i];
    }
    auto [kx, gradh] = sph::veDefGradhJLoop(0, K, box, neighbors.data(), neighborsCount, x.data(), y.data(), z.data(),
                                            h.data(), m.data(), wh.data(), whd.data(), xm.data());

    T density = kx * m[0] / xm[0];
    EXPECT_NEAR(density, 3.4662283566584293e1, 8e-7);
    EXPECT_NEAR(gradh, 0.98699067585409861, 5e-7);
    EXPECT_NEAR(kx, 1.0042661134076782, 3e-7);
}

TEST(VeDefGradh, JLoopPBC)
{
    using T = double;

    T sincIndex = 6.0;
    T K         = compute_3d_k(sincIndex);

    std::array<double, lt::size> wh  = lt::createWharmonicTable<double, lt::size>(sincIndex);
    std::array<double, lt::size> whd = lt::createWharmonicDerivativeTable<double, lt::size>(sincIndex);

    // box length in any dimension must be bigger than 4*h for any particle
    // otherwise the PBC evaluation does not select the closest image
    cstone::Box<T> box(0, 10.5, cstone::BoundaryType::periodic);

    // particle 0 has 4 neighbors
    std::vector<cstone::LocalIndex> neighbors{1, 2, 3, 4};
    unsigned                        neighborsCount = 4;

    std::vector<T> x{1.0, 1.1, 1.4, 9.9, 10.4};
    std::vector<T> y{1.1, 1.2, 1.5, 9.8, 10.2};
    std::vector<T> z{1.2, 1.3, 1.6, 9.7, 10.3};
    std::vector<T> h{2.5, 2.51, 2.52, 2.53, 2.54};
    std::vector<T> m{1.1, 1.2, 1.3, 1.4, 1.5};
    std::vector<T> xm = m;
    /* distances of particle 0 to particle j
     *
     *         direct      PBC
     * j = 1  0.173205   0.173205
     * j = 2  0.69282    0.69282
     * j = 3  15.0715    3.1305
     * j = 4  15.9367    2.26495
     */

    T kx;
    std::tie(kx, std::ignore) = sph::veDefGradhJLoop(0, K, box, neighbors.data(), neighborsCount, x.data(), y.data(),
                                                     z.data(), h.data(), m.data(), wh.data(), whd.data(), xm.data());

    EXPECT_NEAR(kx * m[0] / xm[0], 0.17929212174617015, 1e-9);
}
