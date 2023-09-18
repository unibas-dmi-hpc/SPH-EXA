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

#include "sph/hydro_std/density_kern.hpp"
#include "sph/tables.hpp"

using namespace sph;

TEST(Density, JLoop)
{
    using T = double;

    T sincIndex = 6.0;
    T K         = compute_3d_k(sincIndex);

    std::array<double, lt::size> wh  = lt::createWharmonicTable<double, lt::size>(sincIndex);
    std::array<double, lt::size> whd = lt::createWharmonicDerivativeTable<double, lt::size>(sincIndex);

    cstone::Box<T> box(0, 6, cstone::BoundaryType::open);

    // particle 0 has 4 neighbors
    std::vector<cstone::LocalIndex> clist{0};
    std::vector<cstone::LocalIndex> neighbors{1, 2, 3, 4};
    unsigned                        neighborsCount = 4;

    std::vector<T> x{1.0, 2.1, 3.2, 4.3, 5.4};
    std::vector<T> y{1.1, 2.2, 3.3, 4.4, 5.5};
    std::vector<T> z{1.2, 2.3, 3.4, 4.5, 5.6};
    std::vector<T> h{5.0, 5.1, 5.2, 5.3, 5.4};
    std::vector<T> m{1.1, 1.2, 1.3, 1.4, 1.5};

    /* distances of particle zero to particle j
     *
     * j = 1   1.90526
     * j = 2   3.81051
     * j = 3   5.71577
     * j = 4   7.62102
     */

    T rho = densityJLoop(0, K, box, neighbors.data(), neighborsCount, x.data(), y.data(), z.data(), h.data(), m.data(),
                         wh.data(), whd.data());

    EXPECT_NEAR(rho, 0.014286303097548955, 1e-10);
}

TEST(Density, JLoopPBC)
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
    std::vector<cstone::LocalIndex> clist{0};
    std::vector<cstone::LocalIndex> neighbors{1, 2, 3, 4};
    unsigned                        neighborsCount = 4;

    std::vector<T> x{1.0, 1.1, 1.4, 9.9, 10.4};
    std::vector<T> y{1.1, 1.2, 1.5, 9.8, 10.2};
    std::vector<T> z{1.2, 1.3, 1.6, 9.7, 10.3};
    std::vector<T> h{2.5, 2.51, 2.52, 2.53, 2.54};
    std::vector<T> m{1.1, 1.2, 1.3, 1.4, 1.5};

    /* distances of particle 0 to particle j
     *
     *         direct      PBC
     * j = 1  0.173205   0.173205
     * j = 2  0.69282    0.69282
     * j = 3  15.0715    3.1305
     * j = 4  15.9367    2.26495
     */

    T rho = densityJLoop(0, K, box, neighbors.data(), neighborsCount, x.data(), y.data(), z.data(), h.data(), m.data(),
                         wh.data(), whd.data());

    EXPECT_NEAR(rho, 0.17929212174617015, 1e-9);
}
