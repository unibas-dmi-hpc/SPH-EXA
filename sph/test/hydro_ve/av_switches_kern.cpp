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

#include "sph/hydro_ve/av_switches_kern.hpp"
#include "sph/tables.hpp"

using namespace sph;

TEST(AVswitches, JLoop)
{
    using T = double;

    T sincIndex      = 6.0;
    T K              = compute_3d_k(sincIndex);
    T alphamin       = 0.05;
    T alphamax       = 1.0;
    T decay_constant = 0.2;
    T alphai         = alphamax;
    T dt             = 1.5;

    std::array<double, lt::size> wh  = lt::createWharmonicLookupTable<double, lt::size>();
    std::array<double, lt::size> whd = lt::createWharmonicDerivativeLookupTable<double, lt::size>();

    cstone::Box<T> box(0, 6, 0, 6, 0, 6, false, false, false);

    // particle 0 has 4 neighbors
    std::vector<int> neighbors{1, 2, 3, 4};
    int              neighborsCount = 4, i;

    std::vector<T> x{1.0, 1.1, 3.2, 1.3, 2.4};
    std::vector<T> y{1.1, 1.2, 1.3, 4.4, 5.5};
    std::vector<T> z{1.2, 2.3, 1.4, 1.5, 1.6};
    std::vector<T> h{5.0, 5.1, 5.2, 5.3, 5.4};
    std::vector<T> m{1.0, 1.0, 1.0, 1.0, 1.0};
    std::vector<T> c{0.4, 0.5, 0.6, 0.7, 0.8};

    std::vector<T> vx{0.010, -0.020, 0.030, -0.040, 0.050};
    std::vector<T> vy{-0.011, 0.021, -0.031, 0.041, -0.051};
    std::vector<T> vz{0.091, -0.081, 0.071, -0.061, 0.055};

    std::vector<T> c11{0.21, 0.27, 0.10, 0.45, 0.46};
    std::vector<T> c12{-0.22, -0.29, -0.11, -0.44, -0.47};
    std::vector<T> c13{-0.23, -0.31, -0.12, -0.43, -0.48};
    std::vector<T> c22{0.24, 0.32, 0.13, 0.42, 0.49};
    std::vector<T> c23{-0.25, -0.33, -0.14, -0.41, -0.50};
    std::vector<T> c33{0.26, 0.34, 0.15, 0.40, 0.51};

    std::vector<T> xm{m[0] / 1.1, m[1] / 1.2, m[2] / 1.3, m[3] / 1.4, m[4] / 1.5};
    std::vector<T> kx{1.0, 1.5, 2.0, 2.7, 4.0};
    std::vector<T> divv{-0.4, 0.1, 0.2, 0.7, -2.8};

    for (i = 0; i < neighborsCount + 1; i++)
    {
        kx[i] = K * xm[i] / math::pow(h[i], 3);
    }
    /* distances of particle zero to particle j
     *
     * j = 1   1.10905
     * j = 2   2.21811
     * j = 3   3.32716
     * j = 4   4.63465
     */

    // fill with invalid initial value to make sure that the kernel overwrites it instead of add to it
    T alpha = -1;

    // compute gradient for for particle 0
    alpha = AVswitchesJLoop(0,
                            sincIndex,
                            K,
                            box,
                            neighbors.data(),
                            neighborsCount,
                            x.data(),
                            y.data(),
                            z.data(),
                            vx.data(),
                            vy.data(),
                            vz.data(),
                            h.data(),
                            c.data(),
                            c11.data(),
                            c12.data(),
                            c13.data(),
                            c22.data(),
                            c23.data(),
                            c33.data(),
                            wh.data(),
                            whd.data(),
                            kx.data(),
                            xm.data(),
                            divv.data(),
                            dt,
                            alphamin,
                            alphamax,
                            decay_constant,
                            alphai);

    EXPECT_NEAR(alpha, 0.97980576425580013, 1e-10);
}
