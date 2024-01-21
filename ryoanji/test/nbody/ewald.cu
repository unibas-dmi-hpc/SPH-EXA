/*
 * MIT License
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *               2024 University of Basel
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
 * @brief Direct kernel comparison against the CPU
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "dataset.hpp"
#include "ryoanji/nbody/ewald.cuh"

using namespace cstone;
using namespace ryoanji;

TEST(Ewald, MatchCpu)
{
    using T                = double;
    using KeyType          = uint64_t;
    using MultipoleType    = CartesianQuadrupole<float>;
    using SourceCenterType = util::array<T, 4>;

    /// Test input ************************
    size_t         numBodies        = 100;
    T              G                = 1.5;
    double         lCut             = 2.6;
    double         hCut             = 2.8;
    double         alpha_scale      = 2.0;
    int            numReplicaShells = 1;
    T              coordMax         = 3.0;
    cstone::Box<T> box(-coordMax, coordMax, cstone::BoundaryType::periodic);

    std::vector<T> x(numBodies), y(numBodies), z(numBodies), m(numBodies), h(numBodies);
    ryoanji::makeCubeBodies(x.data(), y.data(), z.data(), m.data(), h.data(), numBodies, coordMax);
    ///**************************************

    MultipoleType rootMultipole;
    auto          centerMass = massCenter<T>(x.data(), y.data(), z.data(), m.data(), 0, numBodies);
    P2M(x.data(), y.data(), z.data(), m.data(), 0, numBodies, centerMass, rootMultipole);

    // upload to device
    thrust::device_vector<T> d_x = x, d_y = y, d_z = z, d_m = m, d_h = h;
    thrust::device_vector<T> p(numBodies), ax(numBodies), ay(numBodies), az(numBodies);

    T utot = 0;
    computeGravityEwaldGpu(makeVec3(centerMass), rootMultipole, 0, numBodies, rawPtr(d_x), rawPtr(d_y), rawPtr(d_z),
                           rawPtr(d_m), box, G, rawPtr(p), rawPtr(ax), rawPtr(ay), rawPtr(az), &utot, numReplicaShells,
                           lCut, hCut, alpha_scale);

    T              utotRef = 0;
    std::vector<T> refP(numBodies), refAx(numBodies), refAy(numBodies), refAz(numBodies);
    computeGravityEwald(makeVec3(centerMass), rootMultipole, 0, numBodies, x.data(), y.data(), z.data(), m.data(), box,
                        G, refP.data(), refAx.data(), refAy.data(), refAz.data(), &utotRef, numReplicaShells, lCut,
                        hCut, alpha_scale);

    // download body accelerations
    thrust::host_vector<T> h_p = p, h_ax = ax, h_ay = ay, h_az = az;

    EXPECT_NEAR(utotRef, utot, 1e-7);

    for (int i = 0; i < numBodies; ++i)
    {
        EXPECT_NEAR(h_p[i], refP[i], 1e-6);
        EXPECT_NEAR(h_ax[i], refAx[i], 1e-6);
        EXPECT_NEAR(h_ay[i], refAy[i], 1e-6);
        EXPECT_NEAR(h_az[i], refAz[i], 1e-6);
    }
}
