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

#include "sph/hydro_std/iad_kern.hpp"
#include "sph/tables.hpp"

using namespace sph;

TEST(IAD, JLoop)
{
    using T = double;

    T sincIndex = 6.0;
    T K         = compute_3d_k(sincIndex);

    std::array<double, lt::size> wh  = lt::createWharmonicLookupTable<double, lt::size>();
    std::array<double, lt::size> whd = lt::createWharmonicDerivativeLookupTable<double, lt::size>();

    cstone::Box<T> box(0, 6, 0, 6, 0, 6, false, false, false);

    // particle 0 has 4 neighbors
    std::vector<int> neighbors{1, 2, 3, 4};
    int              neighborsCount = 4;

    std::vector<T> x{1.0, 1.1, 3.2, 1.3, 2.4};
    std::vector<T> y{1.1, 1.2, 1.3, 4.4, 5.5};
    std::vector<T> z{1.2, 2.3, 1.4, 1.5, 1.6};
    std::vector<T> h{5.0, 5.1, 5.2, 5.3, 5.4};
    std::vector<T> m{1.1, 1.2, 1.3, 1.4, 1.5};
    std::vector<T> rho{0.014, 0.015, 0.016, 0.017, 0.018};

    /* distances of particle zero to particle j
     *
     * j = 1   1.10905
     * j = 2   2.21811
     * j = 3   3.32716
     * j = 4   4.63465
     */

    // fill with invalid initial value to make sure that the kernel overwrites it instead of add to it
    std::vector<T> iad(6, -1);

    // compute the 6 tensor components for particle 0
    IADJLoopSTD(0,
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
                rho.data(),
                wh.data(),
                whd.data(),
                &iad[0],
                &iad[1],
                &iad[2],
                &iad[3],
                &iad[4],
                &iad[5]);

    EXPECT_NEAR(iad[0], 0.68826690705820426, 1e-10);
    EXPECT_NEAR(iad[1], -0.12963692749098227, 1e-10);
    EXPECT_NEAR(iad[2], -0.20435302529035185, 1e-10);
    EXPECT_NEAR(iad[3], 0.39616100615949118, 1e-10);
    EXPECT_NEAR(iad[4], -0.16797800818772629, 1e-10);
    EXPECT_NEAR(iad[5], 1.9055087808073545, 1e-10);
}

TEST(IAD, JLoopPBC)
{
    using T = double;

    T sincIndex = 6.0;
    T K         = compute_3d_k(sincIndex);

    std::array<double, lt::size> wh  = lt::createWharmonicLookupTable<double, lt::size>();
    std::array<double, lt::size> whd = lt::createWharmonicDerivativeLookupTable<double, lt::size>();

    // box length in any dimension must be bigger than 4*h for any particle
    // otherwise the PBC evaluation does not select the closest image
    cstone::Box<T> box(0, 10.5, 0, 10.5, 0, 10.5, true, true, true);

    // particle 0 has 4 neighbors
    std::vector<int> neighbors{1, 2, 3, 4};
    int              neighborsCount = 4;

    std::vector<T> x{1.0, 1.1, 3.2, 1.3, 9.4};
    std::vector<T> y{1.1, 1.2, 1.3, 8.4, 9.5};
    std::vector<T> z{1.2, 2.3, 1.4, 1.5, 9.6};
    std::vector<T> h{2.5, 2.51, 2.52, 2.53, 2.54};
    std::vector<T> m{1.1, 1.2, 1.3, 1.4, 1.5};
    std::vector<T> rho{0.014, 0.015, 0.016, 0.017, 0.018};

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

    IADJLoopSTD(0,
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
                rho.data(),
                wh.data(),
                whd.data(),
                &iad[0],
                &iad[1],
                &iad[2],
                &iad[3],
                &iad[4],
                &iad[5]);

    EXPECT_NEAR(iad[0], 0.42970014180599519, 1e-10);
    EXPECT_NEAR(iad[1], -0.2304555811353339, 1e-10);
    EXPECT_NEAR(iad[2], -0.052317231832885822, 1e-10);
    EXPECT_NEAR(iad[3], 2.8861688071845268, 1e-10);
    EXPECT_NEAR(iad[4], -0.23251632520430554, 1e-10);
    EXPECT_NEAR(iad[5], 0.36028770403046995, 1e-10);
}
