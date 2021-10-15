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

#include <iomanip>

#include "gtest/gtest.h"

#include "cstone/gravity/multipole.hpp"
#include "cstone/sfc/box.hpp"
#include "coord_samples/random.hpp"

using namespace cstone;

//! @brief Tests direct particle-to-particle gravity interactions
TEST(Gravity, P2P)
{
    using T = double;

    T eps2 = 0.05 * 0.05;

    // target
    T x = 1;
    T y = 1;
    T z = 1;

    // source
    T xs[2] = {2, -2};
    T ys[2] = {2, -2};
    T zs[2] = {2, -2};
    T m[2]  = {1, 1};

    T xacc = 0;
    T yacc = 0;
    T zacc = 0;

    particle2particle(x, y, z, xs, ys, zs, m, 2, eps2, &xacc, &yacc, &zacc);

    EXPECT_DOUBLE_EQ(xacc, 0.17082940372214045);
    EXPECT_DOUBLE_EQ(yacc, 0.17082940372214045);
    EXPECT_DOUBLE_EQ(zacc, 0.17082940372214045);
}


/*! @brief Tests the gravity interaction of a multipole with a target particle
 *
 * The gravity on the target particle is first evaluated with the direct P2P sum as a reference.
 * This is compared to the gravity on the target particle that arises from the M2P operation.
 */
TEST(Gravity, M2P)
{
    using T = double;

    Box<T> box(-1, 1);
    LocalParticleIndex numParticles = 100;

    RandomCoordinates<T, SfcKind<unsigned>> coordinates(numParticles, box);

    const T* x = coordinates.x().data();
    const T* y = coordinates.y().data();
    const T* z = coordinates.z().data();

    std::vector<T> masses(numParticles);
    std::generate(begin(masses), end(masses), drand48);

    GravityMultipole<T> multipole = particle2Multipole<T>(x, y, z, masses.data(), numParticles);

    // target particle coordinates
    std::array<T, 3> target    = {-8, 0, 0};

    // reference direct gravity on target
    std::array<T, 3> accDirect = {0, 0, 0};
    particle2particle(target[0], target[1], target[2], x, y, z, masses.data(), numParticles, 0.0,
                      &accDirect[0], &accDirect[1], &accDirect[2]);

    // approximate gravity with multipole interaction
    std::array<T, 3> accApprox = {0, 0, 0};
    multipole2particle(target[0], target[1], target[2], multipole, &accApprox[0], &accApprox[1], &accApprox[2]);

    //std::cout << std::fixed;
    //std::cout.precision(8);
    //std::cout << "direct: " << accDirect[0] << " " << accDirect[1] << " " << accDirect[2] << std::endl;
    //std::cout << "approx: " << accApprox[0] << " " << accApprox[1] << " " << accApprox[2] << std::endl;

    EXPECT_TRUE(std::abs(accApprox[0] - accDirect[0]) < 1e-3);
    EXPECT_TRUE(std::abs(accApprox[1] - accDirect[1]) < 1e-3);
    EXPECT_TRUE(std::abs(accApprox[2] - accDirect[2]) < 1e-3);

    EXPECT_DOUBLE_EQ(accApprox[0], 0.74358243303934313);
    EXPECT_DOUBLE_EQ(accApprox[1], 9.1306187450872109e-05);
    EXPECT_DOUBLE_EQ(accApprox[2], 0.0095252528595820823);
}

/*! @brief tests aggregation of multipoles into a composite multipole
 *
 * The reference multipole is directly constructed from all particles,
 * while the subcell multipoles are constructed from 1/8th of the particles each.
 * The subcells are then aggregated with the M2M operation and compared to the reference.
 */
TEST(Gravity, M2M)
{
    using T = double;

    Box<T> box(-1, 1);
    LocalParticleIndex numParticles = 160;

    RandomCoordinates<T, SfcKind<unsigned>> coordinates(numParticles, box);

    const T* x = coordinates.x().data();
    const T* y = coordinates.y().data();
    const T* z = coordinates.z().data();

    std::vector<T> masses(numParticles);
    std::generate(begin(masses), end(masses), drand48);

    // reference directly constructed from particles
    GravityMultipole<T> reference = particle2Multipole<T>(x, y, z, masses.data(), numParticles);

    LocalParticleIndex eighth = numParticles / 8;
    GravityMultipole<T> sc[8];
    for (int i = 0; i < 8; ++i)
    {
        sc[i] = particle2Multipole<T>(x + i*eighth, y + i*eighth, z + i*eighth, masses.data() + i*eighth, eighth);
    }

    // aggregate subcell multipoles
    GravityMultipole<T> composite = multipole2multipole(sc[0], sc[1], sc[2], sc[3], sc[4], sc[5], sc[6], sc[7]);

    EXPECT_TRUE(std::abs(reference.mass - composite.mass) < 1e-10);
    EXPECT_TRUE(std::abs(reference.xcm  - composite.xcm ) < 1e-10);
    EXPECT_TRUE(std::abs(reference.ycm  - composite.ycm ) < 1e-10);
    EXPECT_TRUE(std::abs(reference.zcm  - composite.zcm ) < 1e-10);
    EXPECT_TRUE(std::abs(reference.qxx  - composite.qxx ) < 1e-10);
    EXPECT_TRUE(std::abs(reference.qxy  - composite.qxy ) < 1e-10);
    EXPECT_TRUE(std::abs(reference.qxz  - composite.qxz ) < 1e-10);
    EXPECT_TRUE(std::abs(reference.qyy  - composite.qyy ) < 1e-10);
    EXPECT_TRUE(std::abs(reference.qyz  - composite.qyz ) < 1e-10);
    EXPECT_TRUE(std::abs(reference.qzz  - composite.qzz ) < 1e-10);
}
