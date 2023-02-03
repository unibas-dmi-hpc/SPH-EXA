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
 * @brief Compare and test different multipole approximations
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "cstone/focus/source_center.hpp"

#include "dataset.hpp"
#include "ryoanji/nbody/cartesian_qpole.hpp"
#include "ryoanji/nbody/kernel.hpp"

using namespace ryoanji;

//! @brief Tests direct particle-to-particle gravity interactions with mass softening
TEST(Gravity, P2PmsoftBase)
{
    using T = double;

    cstone::Vec3<T> target{1, 1, 1};
    T               h = std::sqrt(3) / 2 - 0.001;

    cstone::Vec3<T> source1{2, 2, 2}, source2{-2, -2, -2};

    cstone::Vec4<T> acc{0, 0, 0, 0};
    acc = P2P(acc, target, source1, 1.0, h, h);
    acc = P2P(acc, target, source2, 1.0, h, h);

    // h too small to trigger softening, so results should match the non-softened numbers
    EXPECT_NEAR(acc[0], -0.76980035891950138, 1e-10);
    EXPECT_NEAR(acc[1], 0.17106674642655587, 1e-10);
    EXPECT_NEAR(acc[2], 0.17106674642655587, 1e-10);
    EXPECT_NEAR(acc[3], 0.17106674642655587, 1e-10);
}

//! @brief Tests direct particle-to-particle gravity interactions with mass softening
TEST(Gravity, P2PmsoftH)
{
    using T = double;

    cstone::Vec3<T> target{1, 1, 1};
    T               h = std::sqrt(3) / 2 + 0.001;

    cstone::Vec3<T> source1{2, 2, 2}, source2{-2, -2, -2};

    cstone::Vec4<T> acc{0, 0, 0, 0};
    acc = P2P(acc, target, source1, 1.0, h, h);
    acc = P2P(acc, target, source2, 1.0, h, h);

    EXPECT_NEAR(acc[0], -0.7678049688481372, 1e-10);
    EXPECT_NEAR(acc[1], 0.1704016164027678, 1e-10);
    EXPECT_NEAR(acc[2], 0.1704016164027678, 1e-10);
    EXPECT_NEAR(acc[3], 0.1704016164027678, 1e-10);
}

//! @brief The traversal code relies on P2P self interaction being zero
TEST(Gravity, P2PselfInteraction)
{
    using T = double;

    Vec3<T> pos_i{0, 0, 0};
    T       h = 0.1;
    Vec4<T> acc{0, 0, 0, 0};

    auto self = P2P(acc, pos_i, pos_i, 1.0, h, h);
    EXPECT_EQ(self, acc);
}

TEST(Multipole, P2M)
{
    int numBodies = 1023;

    std::vector<double> x(numBodies);
    std::vector<double> y(numBodies);
    std::vector<double> z(numBodies);
    std::vector<double> m(numBodies);
    std::vector<double> h(numBodies);

    ryoanji::makeCubeBodies(x.data(), y.data(), z.data(), m.data(), h.data(), numBodies);

    CartesianQuadrupole<double>      cartesianQuadrupole;
    cstone::SourceCenterType<double> csCenter =
        cstone::massCenter<double>(x.data(), y.data(), z.data(), m.data(), 0, numBodies);
    P2M(x.data(), y.data(), z.data(), m.data(), 0, numBodies, csCenter, cartesianQuadrupole);

    Vec4<double> centerMass = ryoanji::setCenter(0, numBodies, x.data(), y.data(), z.data(), m.data());

    ryoanji::SphericalMultipole<double, 4> multipole;
    std::fill(multipole.begin(), multipole.end(), 0.0);

    ryoanji::P2M(x.data(), y.data(), z.data(), m.data(), 0, numBodies, centerMass, multipole);

    EXPECT_NEAR(multipole[0], cartesianQuadrupole[Cqi::mass], 1e-6);

    EXPECT_NEAR(centerMass[0], csCenter[0], 1e-6);
    EXPECT_NEAR(centerMass[1], csCenter[1], 1e-6);
    EXPECT_NEAR(centerMass[2], csCenter[2], 1e-6);
    EXPECT_NEAR(centerMass[3], cartesianQuadrupole[Cqi::mass], 1e-6);

    // compare M2P results on a test target
    {
        Vec3<double> testTarget{-8, -8, -8};

        Vec4<double> accM2P{0, 0, 0, 0};
        accM2P = ryoanji::M2P(accM2P, testTarget, util::makeVec3(centerMass), multipole);
        // printf("test acceleration: %f %f %f %f\n", acc[0], acc[1], acc[2], acc[3]);

        // cstone is less precise
        // float ax = 0;
        // float ay = 0;
        // float az = 0;
        // cstone::multipole2particle(
        //    testTarget[0], testTarget[1], testTarget[2], cstoneMultipole, eps2, &ax, &ay, &az);
        // printf("cstone test acceleration: %f %f %f\n", ax, ay, az);

        Vec4<double> accP2P{0, 0, 0, 0};
        for (int i = 0; i < numBodies; ++i)
        {
            accP2P = P2P(accP2P, testTarget, Vec3<double>{x[i], y[i], z[i]}, m[i], 0.0, 0.0);
        }
        // printf("direct acceleration: %f %f %f\n", axd, ayd, azd);

        // compare ryoanji against the direct sum reference
        EXPECT_NEAR(accM2P[0], accP2P[0], 3e-5);
        EXPECT_NEAR(accM2P[1], accP2P[1], 1e-5);
        EXPECT_NEAR(accM2P[2], accP2P[2], 1e-5);
        EXPECT_NEAR(accM2P[3], accP2P[3], 1e-5);
    }
}
