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

#include "sph/hydro_ve/iad_kern.hpp"
#include "sph/tables.hpp"
#include "../../../main/src/io/file_utils.hpp"

using namespace sph;

TEST(IAD, JLoop)
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

    std::fill(m.begin(), m.end(), mpart);

    for (i = 0; i < neighborsCount + 1; i++)
    {
        xm[i] = mpart / rho0[i];
        kx[i] = K * xm[i] / std::pow(h[i], 3);
    }

    // fill with invalid initial value to make sure that the kernel overwrites it instead of add to it
    std::vector<T> iad(6, -1);

    // compute the 6 tensor components for particle 0
    IADJLoop(0, K, box, neighbors.data(), neighborsCount, x.data(), y.data(), z.data(), h.data(), wh.data(), whd.data(),
             xm.data(), kx.data(), &iad[0], &iad[1], &iad[2], &iad[3], &iad[4], &iad[5]);

    EXPECT_NEAR(iad[0], 1.9296619855715329e-18, 1e-10);
    EXPECT_NEAR(iad[1], -1.7838691836843698e-20, 1e-10);
    EXPECT_NEAR(iad[2], -1.2892885646884301e-20, 1e-10);
    EXPECT_NEAR(iad[3], 1.9482845913025683e-18, 1e-10);
    EXPECT_NEAR(iad[4], 1.635410357476855e-20, 1e-10);
    EXPECT_NEAR(iad[5], 1.9246939006338132e-18, 1e-10);
}

TEST(IAD, JLoopPBC)
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

    std::vector<T> x{1.0, 1.1, 3.2, 1.3, 9.4};
    std::vector<T> y{1.1, 1.2, 1.3, 8.4, 9.5};
    std::vector<T> z{1.2, 2.3, 1.4, 1.5, 9.6};
    std::vector<T> h{2.5, 2.51, 2.52, 2.53, 2.54};
    std::vector<T> m{1.1, 1.2, 1.3, 1.4, 1.5};
    std::vector<T> xm{m[0] / 0.014, m[1] / 0.015, m[2] / 0.016, m[3] / 0.017, m[4] / 0.018};
    std::vector<T> kx{1.0, 1.0, 1.0, 1.0, 1.0};

    /* distances of particle 0 to particle j
     *
     *          PBC
     * j = 1    1.10905
     * j = 2    2.21811
     * j = 3    3.22800
     * j = 4    3.63731
     */

    // fill with invalid initial value to make sure that the kernel overwrites it instead of add to it
    std::vector<T> iad(6, -1);

    IADJLoop(0, K, box, neighbors.data(), neighborsCount, x.data(), y.data(), z.data(), h.data(), wh.data(), whd.data(),
             xm.data(), kx.data(), &iad[0], &iad[1], &iad[2], &iad[3], &iad[4], &iad[5]);

    EXPECT_NEAR(iad[0], 0.42970014305820264, 3e-7);
    EXPECT_NEAR(iad[1], -0.23045558110767411, 3e-7);
    EXPECT_NEAR(iad[2], -0.052317231995050187, 3e-7);
    EXPECT_NEAR(iad[3], 2.886168807109148, 3e-7);
    EXPECT_NEAR(iad[4], -0.2325163252006715, 3e-7);
    EXPECT_NEAR(iad[5], 0.36028770439708135, 3e-7);
}
