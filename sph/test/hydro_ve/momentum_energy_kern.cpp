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
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "gtest/gtest.h"

#include "sph/hydro_ve/momentum_energy_kern.hpp"
#include "sph/tables.hpp"

using namespace sph;

TEST(MomentumEnergy, JLoop)
{
    using T = double;

    T sincIndex = 6.0;
    T K         = compute_3d_k(sincIndex);
    T Atmin     = 0.1;
    T Atmax     = 0.2;
    T ramp      = 1.0 / (Atmax - Atmin);

    std::array<double, lt::size> wh  = lt::createWharmonicLookupTable<double, lt::size>();
    std::array<double, lt::size> whd = lt::createWharmonicDerivativeLookupTable<double, lt::size>();

    cstone::Box<T> box(0, 6, cstone::BoundaryType::open);

    // particle 0 has 4 neighbors
    std::vector<cstone::LocalIndex> neighbors;//{1, 2, 3, 4};
    unsigned                        neighborsCount = 4, i;
    for (i = 1; i < neighborsCount + 1; i++) {
      neighbors[i] = i;
    }

    std::vector<T> x{1.0, 1.1, 3.2, 1.3, 2.4};
    std::vector<T> y{1.1, 1.2, 1.3, 4.4, 5.5};
    std::vector<T> z{1.2, 2.3, 1.4, 1.5, 1.6};
    std::vector<T> h{5.0, 5.1, 5.2, 5.3, 5.4};
    std::vector<T> m{1.0, 1.0, 1.0, 1.0, 1.0};
    std::vector<T> gradh{1.25, 1., 0.8, 1.1, 0.51};

    std::vector<T> vx{0.010, -0.020, 0.030, -0.040, 0.050};
    std::vector<T> vy{-0.011, 0.021, -0.031, 0.041, -0.051};
    std::vector<T> vz{0.091, -0.081, 0.071, -0.061, 0.055};

    std::vector<T> c{0.4, 0.5, 0.6, 0.7, 0.8};
    std::vector<T> p{0.2, 0.3, 0.4, 0.5, 0.6};

    std::vector<T> alpha{1.0, 0.05, 0.3, 0.5, 0.3};

    std::vector<T> c11{0.21, 0.27, 0.10, 0.45, 0.46};
    std::vector<T> c12{-0.22, -0.29, -0.11, -0.44, -0.47};
    std::vector<T> c13{-0.23, -0.31, -0.12, -0.43, -0.48};
    std::vector<T> c22{0.24, 0.32, 0.13, 0.42, 0.49};
    std::vector<T> c23{-0.25, -0.33, -0.14, -0.41, -0.50};
    std::vector<T> c33{0.26, 0.34, 0.15, 0.40, 0.51};

    std::vector<T> dvxdx{0.21, 0.27, 0.10, 0.45, 0.46};
    std::vector<T> dvxdy{-0.22, -0.29, -0.11, -0.44, -0.47};
    std::vector<T> dvxdz{-0.23, -0.31, -0.12, -0.43, -0.48};
    std::vector<T> dvydx{0.24, 0.32, 0.13, 0.42, 0.49};
    std::vector<T> dvydy{-0.25, -0.33, -0.14, -0.41, -0.50};
    std::vector<T> dvydz{0.26, 0.34, 0.15, 0.40, 0.51};
    std::vector<T> dvzdx{0.24, 0.32, 0.13, 0.42, 0.49};
    std::vector<T> dvzdy{-0.25, -0.33, -0.14, -0.41, -0.50};
    std::vector<T> dvzdz{0.26, 0.34, 0.15, 0.40, 0.51};

    std::vector<T> xm{m[0] / 1.1, m[1] / 1.2, m[2] / 1.3, m[3] / 1.4, m[4] / 1.5};

    std::vector<T> kx{1.0, 1.5, 2.0, 2.7, 4.0};

/*

    std::ifstream in("UT_IC.d");
    std::vector<std::vector<T> > v;
    if (in) {
        std::string line;
        while (std::getline(in, line)) {
            v.push_back(std::vector<T>());

            // Break down the row into column values
            std::stringstream split(line);
            T value;

            while (split >> value)
                v.back().push_back(value);
        }
    }

    std::fill(m.begin(), m.end(), v[0][0]);
    //dt = v[0][1];
    for (i = 1; i < v.size(); i++) {
        x[i]        = v[i][0];
        y[i]        = v[i][1];
        z[i]        = v[i][2];
        vx[i]       = v[i][3];
        vy[i]       = v[i][4];
        vz[i]       = v[i][5];
        h[i]        = v[i][6];
        c[i]        = v[i][7];
        c11[i]      = v[i][8];
        c12[i]      = v[i][9];
        c13[i]      = v[i][10];
        c22[i]      = v[i][11];
        c23[i]      = v[i][12];
        c33[i]      = v[i][13];
        p[i]        = v[i][14];
        gradh[i]    = v[i][15];
        //rho0[i]     = v[i][16];
        //sumwhro0[i] = v[i][17];
        xm[i]       = v[i][18];
        dvxdx[i]    = v[i][19];
        dvxdy[i]    = v[i][20];
        dvxdz[i]    = v[i][21];
        dvydx[i]    = v[i][22];
        dvydy[i]    = v[i][23];
        dvydz[i]    = v[i][24];
        dvzdx[i]    = v[i][25];
        dvzdy[i]    = v[i][26];
        dvzdz[i]    = v[i][27];
        alpha[i]    = v[i][28];
        //u[i]        = v[i][29];
        //divv[i]     = v[i][30];
    }
    */

    for (i = 0; i < neighborsCount + 1; i++)
    {
        kx[i] = K * xm[i] / math::pow(h[i], 3);
    }

    std::vector<T> prho(p.size());
    for (size_t k = 0; k < prho.size(); ++k)
    {
        prho[k] = p[k] / (kx[k] * m[k] * m[k] * gradh[k]);
    }

    /* distances of particle zero to particle j
     *
     * j = 1   1.10905
     * j = 2   2.21811
     * j = 3   3.32716
     * j = 4   4.63465
     */
    // fill with invalid initial value to make sure that the kernel overwrites it instead of add to it
    T du         = -1;
    T grad_Px    = -1;
    T grad_Py    = -1;
    T grad_Pz    = -1;
    T maxvsignal = -1;

    // compute gradient for for particle 0
    momentumAndEnergyJLoop(0, sincIndex, K, box, neighbors.data(), neighborsCount, x.data(), y.data(), z.data(),
                           vx.data(), vy.data(), vz.data(), h.data(), m.data(), prho.data(), c.data(), c11.data(),
                           c12.data(), c13.data(), c22.data(), c23.data(), c33.data(), Atmin, Atmax, ramp, wh.data(),
                           whd.data(), kx.data(), xm.data(), alpha.data(), dvxdx.data(), dvxdy.data(), dvxdz.data(),
                           dvydx.data(), dvydy.data(), dvydz.data(), dvzdx.data(), dvzdy.data(), dvzdz.data(),
                           &grad_Px, &grad_Py, &grad_Pz, &du, &maxvsignal);

    EXPECT_NEAR(grad_Px, 4.6852624676440924e-1, 1e-10);
    EXPECT_NEAR(grad_Py, -8.2810161944474575e-2, 1e-10);
    EXPECT_NEAR(grad_Pz, 5.209843022360216e-1, 1e-10);
    EXPECT_NEAR(du, -3.8445778269613888e-3, 1e-10);
    EXPECT_NEAR(maxvsignal, 1.4112466829, 1e-10);
}
