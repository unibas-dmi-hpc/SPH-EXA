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
 * @brief Direct kernel comparison against the CPU
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "dataset.hpp"
#include "ryoanji/gpu_config.h"
#include "ryoanji/direct.cuh"
#include "ryoanji/cpu/treewalk.hpp"

using namespace ryoanji;

template<class T>
std::vector<Vec4<T>> cpuReference(const std::vector<Vec4<T>>& bodies)
{
    size_t numBodies = bodies.size();

    std::vector<double> x(numBodies);
    std::vector<double> y(numBodies);
    std::vector<double> z(numBodies);
    std::vector<double> h(numBodies, 0.0);
    std::vector<double> m(numBodies);

    for (size_t i = 0; i < numBodies; ++i)
    {
        x[i] = bodies[i][0];
        y[i] = bodies[i][1];
        z[i] = bodies[i][2];
        m[i] = bodies[i][3];
    }

    std::vector<double> ax(numBodies);
    std::vector<double> ay(numBodies);
    std::vector<double> az(numBodies);
    std::vector<double> pot(numBodies);

    float G = 1.0;

    cstone::directSum(x.data(), y.data(), z.data(), h.data(), m.data(), numBodies, G,
                      ax.data(), ay.data(), az.data(), pot.data());

    std::vector<Vec4<T>> acc(numBodies, Vec4<T>{0, 0, 0, 0});

    for (size_t i = 0; i < numBodies; ++i)
    {
        acc[i] = Vec4<T>{T(pot[i]), T(ax[i]), T(ay[i]), T(az[i])};
    }

    return acc;
}

TEST(DirectSum, MatchCpu)
{
    using T = float;
    int npOnEdge  = 10;
    int numBodies = npOnEdge * npOnEdge * npOnEdge;

    // the CPU reference uses mass softening, while the GPU P2P kernel still uses plummer softening
    // so the only way to compare is without softening in both versions and make sure that
    // particles are not on top of each other
    float eps = 0.0;

    std::vector<Vec4<T>> h_bodies(numBodies);
    ryoanji::makeGridBodies(h_bodies.data(), npOnEdge, 0.5);

    // upload to device
    thrust::device_vector<Vec4<T>> bodyPos = h_bodies;
    thrust::device_vector<Vec4<T>> bodyAcc(numBodies, Vec4<T>{0, 0, 0, 0});

    directSum(numBodies, rawPtr(bodyPos.data()), rawPtr(bodyAcc.data()), eps);

    // download body accelerations
    thrust::host_vector<Vec4<T>> h_acc = bodyAcc;

    auto refAcc = cpuReference(h_bodies);

    for (int i = 0; i < numBodies; ++i)
    {
        Vec3<T> ref   = {refAcc[i][1], refAcc[i][2], refAcc[i][3]};
        Vec3<T> probe = {h_acc[i][1], h_acc[i][2], h_acc[i][3]};

        EXPECT_NEAR(std::sqrt(norm2(ref - probe) / norm2(probe)), 0, 1e-6);
        // the potential
        EXPECT_NEAR(refAcc[i][0], h_acc[i][0], 1e-6);

        // printf("%f %f %f\n", ref[1], ref[2], ref[3]);
        // printf("%f %f %f\n", probe[1], probe[2], probe[3]);
    }
}
