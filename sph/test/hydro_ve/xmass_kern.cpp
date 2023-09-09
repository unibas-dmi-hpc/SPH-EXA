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

#include "sph/hydro_ve/xmass_kern.hpp"
#include "sph/tables.hpp"
#include "../../../main/src/io/file_utils.hpp"

using namespace sph;

TEST(xmass, JLoop)
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

    std::vector<T*> fields{x.data(),     y.data(),     z.data(),     vx.data(),    vy.data(),    vz.data(),
                           h.data(),     c.data(),     c11.data(),   c12.data(),   c13.data(),   c22.data(),
                           c23.data(),   c33.data(),   p.data(),     gradh.data(), rho0.data(),  sumwhrho0.data(),
                           sumwh.data(), dvxdx.data(), dvxdy.data(), dvxdz.data(), dvydx.data(), dvydy.data(),
                           dvydz.data(), dvzdx.data(), dvzdy.data(), dvzdz.data(), alpha.data(), u.data(),
                           divv.data()};

    sphexa::fileutils::readAscii("example_data.txt", npart, fields);

    std::fill(m.begin(), m.end(), mpart);

    T xmass = xmassJLoop(0, K, box, neighbors.data(), neighborsCount, x.data(), y.data(), z.data(), h.data(), m.data(),
                         wh.data(), whd.data());
    T rho0i = m[0] / xmass;

    EXPECT_NEAR(rho0i, 34.515038498081417, 7.33e-7);
    EXPECT_NEAR(xmass, m[0] / rho0i, 1e-10);
    EXPECT_NEAR(xmass, m[0] / rho0[0], m[0] / rho0[0] * 1.e-7);
}
