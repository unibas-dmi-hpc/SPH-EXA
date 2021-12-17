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
 * @brief Integration test between gravity multipole upsweep and tree walk
 *
 * @author Sebastian Keller        <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "cstone/gravity/treewalk.hpp"
#include "cstone/gravity/upsweep.hpp"
#include "cstone/sfc/box.hpp"
#include "coord_samples/random.hpp"

using namespace cstone;

TEST(Gravity, TreeWalk)
{
    using T = double;
    using KeyType = uint64_t;

    float G = 1.0;
    unsigned bucketSize = 64;
    Box<T> box(-1, 1);
    LocalParticleIndex numParticles = 10000;

    RandomCoordinates<T, SfcKind<KeyType>> coordinates(numParticles, box);

    const T* x = coordinates.x().data();
    const T* y = coordinates.y().data();
    const T* z = coordinates.z().data();

    std::vector<T> h(numParticles, 0.01);
    std::vector<T> masses(numParticles);
    std::generate(begin(masses), end(masses), drand48);

    // the leaf cells and leaf particle counts
    auto [treeLeaves, counts] = computeOctree(coordinates.particleKeys().data(),
                                              coordinates.particleKeys().data() + numParticles,
                                              bucketSize);

    // fully linked octree, including internal part
    Octree<KeyType> octree;
    octree.update(std::move(treeLeaves));

    // layout[i] is equal to the index in (x,y,z,m) of the first particle in leaf cell with index i
    std::vector<LocalParticleIndex> layout(octree.numLeafNodes() + 1);
    stl::exclusive_scan(counts.begin(), counts.end() + 1, layout.begin(), LocalParticleIndex(0));

    std::vector<GravityMultipole<T>> multipoles(octree.numTreeNodes());
    computeMultipoles(octree, layout, x, y, z, masses.data(), multipoles.data());

    T totalMass = std::accumulate(masses.begin(), masses.end(), 0.0);
    EXPECT_TRUE(std::abs(totalMass - multipoles[0].mass) < 1e-6);

    std::vector<T> ax(numParticles, 0);
    std::vector<T> ay(numParticles, 0);
    std::vector<T> az(numParticles, 0);
    std::vector<T> potential(numParticles, 0);

    float theta = 0.6;

    computeGravity(octree, multipoles.data(), layout.data(), 0, octree.numLeafNodes(),
                   x, y, z, h.data(), masses.data(), box, theta, G, ax.data(), ay.data(), az.data(),
                   potential.data());

    // test version that computes total grav energy only instead of per particle
    {
        double egravTot = 0.0;
        for (size_t i = 0; i < numParticles; ++i)
        {
            egravTot += masses[i] * potential[i];
        }
        egravTot *= 0.5;

        std::vector<T> ax2(numParticles, 0);
        std::vector<T> ay2(numParticles, 0);
        std::vector<T> az2(numParticles, 0);
        double egravTot2 = computeGravity(octree, multipoles.data(), layout.data(), 0,
                                          octree.numLeafNodes(), x, y, z,
                                          h.data(), masses.data(), box, theta, G, ax2.data(), ay2.data(), az2.data());
        std::cout << "total gravitational energy: " << egravTot << std::endl;
        EXPECT_NEAR((egravTot-egravTot2)/egravTot, 0, 1e-4);
    }

    // direct sum reference
    std::vector<T> Ax(numParticles, 0);
    std::vector<T> Ay(numParticles, 0);
    std::vector<T> Az(numParticles, 0);
    std::vector<T> potentialReference(numParticles, 0);

    directSum(x, y, z, h.data(), masses.data(), numParticles, G, Ax.data(), Ay.data(), Az.data(),
              potentialReference.data());

    // relative errors
    std::vector<T> delta(numParticles);
    for (LocalParticleIndex i = 0; i < numParticles; ++i)
    {
        T dx = ax[i] - Ax[i];
        T dy = ay[i] - Ay[i];
        T dz = az[i] - Az[i];

        delta[i] = std::sqrt( (dx*dx + dy*dy + dz*dz) / (Ax[i]*Ax[i] + Ay[i]*Ay[i] + Az[i]*Az[i]));

        EXPECT_NEAR((potential[i] - potentialReference[i]) / potentialReference[i], 0, 1e-2);
    }

    // sort errors in ascending order to infer the error distribution
    std::sort(begin(delta), end(delta));

    EXPECT_TRUE(delta[numParticles*0.99] < 2e-3);
    EXPECT_TRUE(delta[numParticles-1] < 2e-2);

    std::cout.precision(10);
    std::cout << "min Error: "       << delta[0] << std::endl;
    // 50% of particles have an error smaller than this
    std::cout << "50th percentile: " << delta[numParticles/2] << std::endl;
    // 90% of particles have an error smaller than this
    std::cout << "10th percentile: " << delta[numParticles*0.9] << std::endl;
    // 99% of particles have an error smaller than this
    std::cout << "1st percentile: "  << delta[numParticles*0.99] << std::endl;
    std::cout << "max Error: "       << delta[numParticles-1] << std::endl;
}
