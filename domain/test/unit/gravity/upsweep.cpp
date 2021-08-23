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
 * @brief gravity multipole upsweep test
 *
 * @author Sebastian Keller        <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "cstone/gravity/upsweep.hpp"
#include "cstone/sfc/box.hpp"
#include "coord_samples/random.hpp"

using namespace cstone;

TEST(Gravity, upsweep)
{
    using T = double;
    using KeyType = uint64_t;

    unsigned bucketSize = 64;
    Box<T> box(-1, 1);
    LocalParticleIndex numParticles = 10000;

    RandomCoordinates<T, SfcKind<KeyType>> coordinates(numParticles, box);

    const T* x = coordinates.x().data();
    const T* y = coordinates.y().data();
    const T* z = coordinates.z().data();

    std::vector<T> masses(numParticles);
    std::generate(begin(masses), end(masses), drand48);

    auto [tree, counts] = computeOctree(coordinates.particleKeys().data(),
                                        coordinates.particleKeys().data() + numParticles,
                                        bucketSize);
    Octree<KeyType> octree;
    octree.update(std::move(tree));

    std::vector<LocalParticleIndex> layout(octree.numLeafNodes() + 1);
    stl::exclusive_scan(counts.begin(), counts.end() + 1, layout.begin(), LocalParticleIndex(0));

    std::vector<GravityMultipole<T>> multipoles(octree.numTreeNodes());
    computeMultipoles(octree, layout, x, y, z, masses.data(), multipoles.data());

    T totalMass = std::accumulate(masses.begin(), masses.end(), 0.0);
    std::cout.precision(10);
    std::cout << multipoles[0].mass << std::endl;
    std::cout << totalMass << std::endl;
    EXPECT_TRUE(std::abs(totalMass - multipoles[0].mass) < 1e-6);
}
