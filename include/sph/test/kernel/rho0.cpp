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
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <vector>

#include "gtest/gtest.h"

#include "sph/kernel/computeRho0.hpp"
#include "sph/lookupTables.hpp"

using namespace sphexa;

TEST(rho0, JLoop)
{
    using T = double;

    T sincIndex = 6.0;
    T K = compute_3d_k(sincIndex);

    std::array<double, lt::size> wh  = lt::createWharmonicLookupTable<double, lt::size>();
    std::array<double, lt::size> whd = lt::createWharmonicDerivativeLookupTable<double, lt::size>();

    cstone::Box<T> box(0, 6, 0, 6, 0, 6, false, false, false);

    // particle 0 has 4 neighbors
    std::vector<int> clist{0};
    std::vector<int> neighbors{1, 2, 3, 4};
    int neighborsCount = 4;

    std::vector<T> x{1.0, 1.1, 3.2, 1.3, 2.4};
    std::vector<T> y{1.1, 1.2, 1.3, 4.4, 5.5};
    std::vector<T> z{1.2, 2.3, 1.4, 1.5, 1.6};
    std::vector<T> h{5.0, 5.1, 5.2, 5.3, 5.4};
    std::vector<T> m{1.0, 1.0, 1.0, 1.0, 1.0};
    std::vector<T> rho{0.014, 0.015, 0.016, 0.017, 0.018};
    std::vector<T> rho0{-1.0, -1.0, -1.0, -1.0, -1.0};
    std::vector<T> wrho0{-1.0, -1.0, -1.0, -1.0, -1.0};

    /* distances of particle zero to particle j
     *
     * j = 1   1.90526
     * j = 2   3.81051
     * j = 3   5.71577
     * j = 4   7.62102
     */
     int i, T sincIndex, T K, const cstone::Box<T>& box, const int* neighbors,
                                                int neighborsCount, const T* x, const T* y, const T* z, const T* h,
                                                const T* m, const T* wh, const T* whd, T* rho0, T* wrho0)
    sph::kernels::rho0JLoop(0,
                               sincIndex,
                               K,
                               box,
                               neighbors.data(),
                               neighborsCount,
                               x.data(),
                               y.data(),
                               z.data(),
                               h.data(),
                               m.data(),
                               wh.data(),
                               whd.data(),
                               rho0.data(),
                               wrho0.data());
    EXPECT_NEAR(rho0[0], 1.8450716246e-2, 1e-10);
    EXPECT_NEAR(wrho0[0], -8.4242274598e-3, 1e-10);
}

TEST(rho0, JLoopPBC)
{
    using T = double;

    T sincIndex = 6.0;
    T K = compute_3d_k(sincIndex);

    std::array<double, lt::size> wh  = lt::createWharmonicLookupTable<double, lt::size>();
    std::array<double, lt::size> whd = lt::createWharmonicDerivativeLookupTable<double, lt::size>();

    // box length in any dimension must be bigger than 4*h for any particle
    // otherwise the PBC evaluation does not select the closest image
    cstone::Box<T> box(0, 10.5, 0, 10.5, 0, 10.5, true, true, true);

    // particle 0 has 4 neighbors
    std::vector<int> clist{0};
    std::vector<int> neighbors{1, 2, 3, 4};
    int neighborsCount = 4;

    std::vector<T> x{1.0, 1.1, 3.2, 1.3, 9.4};
    std::vector<T> y{1.1, 1.2, 1.3, 8.4, 9.5};
    std::vector<T> z{1.2, 2.3, 1.4, 1.5, 9.6};
    std::vector<T> h{2.5, 2.51, 2.52, 2.53, 2.54};
    std::vector<T> m{1.1, 1.2, 1.3, 1.4, 1.5};
    std::vector<T> rho{0.014, 0.015, 0.016, 0.017, 0.018};
    std::vector<T> rho0{-1.0, -1.0, -1.0, -1.0, -1.0};
    std::vector<T> wrho0{-1.0, -1.0, -1.0, -1.0, -1.0};
    /* distances of particle 0 to particle j
     *
     *         direct      PBC
     * j = 1  0.173205   0.173205
     * j = 2  0.69282    0.69282
     * j = 3  15.0715    3.1305
     * j = 4  15.9367    2.26495
     */

    sph::kernels::densityJLoop(0,
                               sincIndex,
                               K,
                               box,
                               neighbors.data(),
                               neighborsCount,
                               x.data(),
                               y.data(),
                               z.data(),
                               h.data(),
                               m.data(),
                               wh.data(),
                               whd.data(),
                               rho0.data(),
                               wrho0.data(),
                               rho.data(),
                               kx.data(),
                               whomega.data());

    EXPECT_NEAR(rho[0], 0.17929212293724384, 1e-10);
    EXPECT_NEAR(rho0[0], 0.014286303130604867, 1e-10);
    EXPECT_NEAR(wrho0[0], 0.014286303130604867, 1e-10);
}
