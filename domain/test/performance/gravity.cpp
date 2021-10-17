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

#include <chrono>

#include "cstone/gravity/treewalk.hpp"
#include "cstone/gravity/upsweep.hpp"
#include "cstone/sfc/box.hpp"
#include "coord_samples/random.hpp"

using namespace cstone;

int main()
{
    using T = float;
    using KeyType = uint64_t;

    unsigned bucketSize             = 64;
    float theta                     = 0.75;
    LocalParticleIndex numParticles = 100000;
    Box<T> box(-1, 1);

    RandomCoordinates<T, SfcKind<KeyType>> coordinates(numParticles, box);

    const T* x = coordinates.x().data();
    const T* y = coordinates.y().data();
    const T* z = coordinates.z().data();

    std::vector<T> h(numParticles, 0.01);
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

    std::vector<T> ax(numParticles, 0);
    std::vector<T> ay(numParticles, 0);
    std::vector<T> az(numParticles, 0);
    std::vector<T> pot(numParticles, 0);

    // direct sum reference
    std::vector<double> Ax(numParticles, 0);
    std::vector<double> Ay(numParticles, 0);
    std::vector<double> Az(numParticles, 0);
    std::vector<double> potRef(numParticles, 0);

    auto t0 = std::chrono::high_resolution_clock::now();
    computeGravity(octree, multipoles.data(), layout.data(), 0, octree.numLeafNodes(),
                   x, y, z, h.data(), masses.data(), box, theta, ax.data(), ay.data(), az.data(),
                   pot.data());
    auto t1 = std::chrono::high_resolution_clock::now();
    float elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "Time elapsed for " << numParticles << " particles: " << elapsed << " s, "
              << double(numParticles) / 1e6 / elapsed << " million particles/second" << std::endl;

    {
        std::vector<double> xd(x, x + numParticles);
        std::vector<double> yd(y, y + numParticles);
        std::vector<double> zd(z, z + numParticles);
        auto t0 = std::chrono::high_resolution_clock::now();
        directSum(xd.data(), yd.data(), zd.data(), h.data(), masses.data(), numParticles,
                  Ax.data(), Ay.data(), Az.data(), potRef.data());
        auto t1 = std::chrono::high_resolution_clock::now();
        float elapsed = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "Time elapsed for direct sum: " << elapsed << " s, "
                  << double(numParticles) / 1e6 / elapsed << " million particles/second" << std::endl;
    }

    std::vector<T> delta(numParticles);
    #pragma omp parallel for schedule(static)
    for (LocalParticleIndex i = 0; i < numParticles; ++i)
    {
        T dx = ax[i] - Ax[i];
        T dy = ay[i] - Ay[i];
        T dz = az[i] - Az[i];

        delta[i] = std::sqrt( (dx*dx + dy*dy + dz*dz) / (Ax[i]*Ax[i] + Ay[i]*Ay[i] + Az[i]*Az[i]));
    }

    std::sort(begin(delta), end(delta));

    std::cout.precision(10);
    std::cout << "min Error: "       << delta[0] << std::endl;
    std::cout << "50th percentile: " << delta[numParticles/2] << std::endl;
    std::cout << "10th percentile: " << delta[numParticles*0.9] << std::endl;
    std::cout << "1st percentile: "  << delta[numParticles*0.99] << std::endl;
    std::cout << "max Error: "       << delta[numParticles-1] << std::endl;
}
