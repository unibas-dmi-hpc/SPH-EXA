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

#include "gtest/gtest.h"

#include "cstone/sfc/box.hpp"
#include "coord_samples/random.hpp"
#include "ryoanji/nbody/traversal_cpu.hpp"
#include "ryoanji/nbody/traversal_pbc_cpu.hpp"
#include "ryoanji/nbody/upsweep_cpu.hpp"

using namespace cstone;
using namespace ryoanji;

TEST(Gravity, TreeWalkPBC)
{
    GTEST_SKIP() << "Skipping TreeWalkPBC";

    using T             = double;
    using KeyType       = uint64_t;
    using MultipoleType = ryoanji::CartesianQuadrupole<T>;

    float          theta      = 1.0;
    float          G          = 1.0;
    unsigned       bucketSize = 64;
    cstone::Box<T> box(-1, 1, cstone::BoundaryType::periodic);
    //cstone::Box<T> box(-1, 1,
    //                   -1, 1,
    //                   -1, 1,
    //                   cstone::BoundaryType::periodic,
    //                   cstone::BoundaryType::open,
    //                   cstone::BoundaryType::open);
    LocalIndex     numParticles = 10000;

    RandomCoordinates<T, SfcKind<KeyType>> coordinates(numParticles, box);

    const T* x = coordinates.x().data();
    const T* y = coordinates.y().data();
    const T* z = coordinates.z().data();

    std::vector<T> h(numParticles, 0.01);
    std::vector<T> masses(numParticles);
    std::generate(begin(masses), end(masses), drand48);

    // the leaf cells and leaf particle counts
    auto [treeLeaves, counts] =
        computeOctree(coordinates.particleKeys().data(), coordinates.particleKeys().data() + numParticles, bucketSize);

    // fully linked octree, including internal part
    OctreeData<KeyType, CpuTag> octree;
    octree.resize(nNodes(treeLeaves));
    updateInternalTree<KeyType>(treeLeaves, octree.data());

    // layout[i] is equal to the index in (x,y,z,m) of the first particle in leaf cell with index i
    std::vector<LocalIndex> layout(octree.numLeafNodes + 1);
    stl::exclusive_scan(counts.begin(), counts.end() + 1, layout.begin(), LocalIndex(0));

    auto toInternal = leafToInternal(octree);

    std::vector<SourceCenterType<T>> centers(octree.numNodes);
    computeLeafMassCenter<T, T, T>(coordinates.x(), coordinates.y(), coordinates.z(), masses, toInternal, layout.data(),
                                   centers.data());
    upsweep(octree.levelRange, octree.childOffsets, centers.data(), CombineSourceCenter<T>{});
    setMac<T, KeyType>(octree.prefixes, centers, 1.0 / theta, box);

    std::vector<MultipoleType> multipoles(octree.numNodes);
    computeLeafMultipoles(x, y, z, masses.data(), toInternal, layout.data(), centers.data(), multipoles.data());
    upsweepMultipoles(octree.levelRange, octree.childOffsets.data(), centers.data(), multipoles.data());
    for (size_t i = 0; i < multipoles.size(); ++i)
    {
        multipoles[i] = ryoanji::normalize(multipoles[i]);
    }

    T totalMass = std::accumulate(masses.begin(), masses.end(), 0.0);
    EXPECT_TRUE(std::abs(totalMass - multipoles[0][ryoanji::Cqi::mass]) < 1e-6);

    std::vector<T> ax(numParticles, 0);
    std::vector<T> ay(numParticles, 0);
    std::vector<T> az(numParticles, 0);

    auto   t0       = std::chrono::high_resolution_clock::now();
    double egravTot = computeGravity(octree.childOffsets.data(), octree.internalToLeaf.data(), octree.numLeafNodes,
                                     centers.data(), multipoles.data(), layout.data(), 0, octree.numLeafNodes, x, y, z,
                                     h.data(), masses.data(), G, ax.data(), ay.data(), az.data());
    auto   t1       = std::chrono::high_resolution_clock::now();
    double elapsed  = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "Time elapsed for " << numParticles << " particles: " << elapsed << " s, "
              << double(numParticles) / 1e6 / elapsed << " million particles/second" << std::endl;

    // direct sum reference
    std::vector<T> Ax(numParticles, 0);
    std::vector<T> Ay(numParticles, 0);
    std::vector<T> Az(numParticles, 0);
    std::vector<T> potentialReference(numParticles, 0);

    t0 = std::chrono::high_resolution_clock::now();
    directSum(x, y, z, h.data(), masses.data(), numParticles, G, Ax.data(), Ay.data(), Az.data(),
              potentialReference.data());
    t1      = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration<double>(t1 - t0).count();
    
    // relative errors
    std::vector<T> delta(numParticles);
    for (LocalIndex i = 0; i < numParticles; ++i)
    {
        T dx = ax[i] - Ax[i];
        T dy = ay[i] - Ay[i];
        T dz = az[i] - Az[i];

        delta[i] = std::sqrt((dx * dx + dy * dy + dz * dz) / (Ax[i] * Ax[i] + Ay[i] * Ay[i] + Az[i] * Az[i]));
    }

    std::cout << "Time elapsed for direct sum: " << elapsed << " s, " << double(numParticles) / 1e6 / elapsed
              << " million particles/second" << std::endl;

    double refPotSum = 0;
    for (LocalIndex i = 0; i < numParticles; ++i)
    {
        refPotSum += potentialReference[i];
    }
    refPotSum *= 0.5;
    if (refPotSum != 0.0)
        EXPECT_NEAR(std::abs(refPotSum - egravTot) / refPotSum, 0, 1e-2);
    else
        EXPECT_NEAR(std::abs(refPotSum - egravTot), 0, 1e-2);


    std::vector<T> thetas;
    thetas.push_back(0.2);
    thetas.push_back(0.4);
    thetas.push_back(0.6);
    thetas.push_back(0.8);
    thetas.push_back(1.0);
    for (size_t theta_idx=0; theta_idx < thetas.size(); theta_idx++)
    {
        theta = thetas[theta_idx];
        setMac<T, KeyType>(octree.prefixes, centers, 1.0 / theta, box);

        for (LocalIndex i = 0; i < numParticles; ++i)
        {
            ax[i] = ay[i] = az[i] = 0.0;
        }

        auto maxNumShells = 8;
        for (int numShells=0; numShells <= maxNumShells; numShells++)
        {

            auto   t0       = std::chrono::high_resolution_clock::now();
            double egravTotPBC = computeGravityPBC(octree.childOffsets.data(), octree.internalToLeaf.data(), octree.numLeafNodes,
                                             centers.data(), multipoles.data(), layout.data(), 0, octree.numLeafNodes, x, y, z,
                                             h.data(), masses.data(), G, ax.data(), ay.data(), az.data(), 
                                             box, numShells, true);
            auto   t1       = std::chrono::high_resolution_clock::now();
            double elapsed  = std::chrono::duration<double>(t1 - t0).count();

            cstone::Vec3<T> atot{0.0, 0.0, 0.0};
            for (LocalIndex i = 0; i < numParticles; ++i)
            {
                atot[0] += ax[i];
                atot[1] += ay[i];
                atot[2] += az[i];
                //printf("%23.15f  %23.15f  %23.15f\n", ax[i], ay[i], az[i]);
            }
            T amag = std::sqrt(atot[0]*atot[0] + atot[1]*atot[1] + atot[2]*atot[2]);
            //std::cout << numShells << "/" << maxNumShells << "] Ref. Pot: "  << refPotSum << "    Est. Pot: " << egravTot << "    Est. Pot. PBC: " << egravTotPBC << "    PBC Time: " << elapsed << "s" << std::endl;
            //std::cout << numShells << "/" << maxNumShells << "] Ref. Pot: "  << refPotSum << "    Est. Pot: " << egravTot << "    Est. Pot. PBC: " << egravTotPBC << "Est. Tot a: " << amag << "    PBC Time: " << elapsed << "s" << std::endl;
            std::cout << numShells << " / " << maxNumShells 
                << "] Ref. Pot: "        << refPotSum 
                << "    Est. Pot: "      << egravTot 
                << "    Est. Pot. PBC: " << egravTotPBC 
                << "    Est. Tot |a|: "  << amag 
                << "    PBC Time: "      << elapsed << "s" 
                << std::endl;
        }

        std::cout << "-------------------------------------------------------------------" << std::endl;
    }

    // sort errors in ascending order to infer the error distribution
    std::sort(begin(delta), end(delta));

    EXPECT_TRUE(delta[numParticles * 0.99] < 3e-3);
    EXPECT_TRUE(delta[numParticles - 1] < 2e-2);

    std::cout.precision(10);
    std::cout << "min Error: " << delta[0] << std::endl;
    // 50% of particles have an error smaller than this
    std::cout << "50th percentile: " << delta[numParticles / 2] << std::endl;
    // 90% of particles have an error smaller than this
    std::cout << "10th percentile: " << delta[numParticles * 0.9] << std::endl;
    // 99% of particles have an error smaller than this
    std::cout << "1st percentile: " << delta[numParticles * 0.99] << std::endl;
    std::cout << "max Error: " << delta[numParticles - 1] << std::endl;
}

TEST(Gravity, TreeWalkEwald)
{
    //GTEST_SKIP() << "Skipping TreeWalkEwald";

    using T             = double;
    using KeyType       = uint64_t;
    using MultipoleType = ryoanji::CartesianQuadrupole<T>;

    float          theta      = 1.0;
    float          G          = 1.0;
    unsigned       bucketSize = 64;
    cstone::Box<T> box(-1, 1, cstone::BoundaryType::periodic);
    //cstone::Box<T> box(-1, 1,
    //                   -1, 1,
    //                   -1, 1,
    //                   cstone::BoundaryType::periodic,
    //                   cstone::BoundaryType::open,
    //                   cstone::BoundaryType::open);
    LocalIndex     numParticles = 10000;

    RandomCoordinates<T, SfcKind<KeyType>> coordinates(numParticles, box);

    const T* x = coordinates.x().data();
    const T* y = coordinates.y().data();
    const T* z = coordinates.z().data();

    std::vector<T> h(numParticles, 0.01);
    std::vector<T> masses(numParticles, 1);
    std::generate(begin(masses), end(masses), drand48);

    // the leaf cells and leaf particle counts
    auto [treeLeaves, counts] =
        computeOctree(coordinates.particleKeys().data(), coordinates.particleKeys().data() + numParticles, bucketSize);

    // fully linked octree, including internal part
    OctreeData<KeyType, CpuTag> octree;
    octree.resize(nNodes(treeLeaves));
    updateInternalTree<KeyType>(treeLeaves, octree.data());

    // layout[i] is equal to the index in (x,y,z,m) of the first particle in leaf cell with index i
    std::vector<LocalIndex> layout(octree.numLeafNodes + 1);
    stl::exclusive_scan(counts.begin(), counts.end() + 1, layout.begin(), LocalIndex(0));

    auto toInternal = leafToInternal(octree);

    std::vector<SourceCenterType<T>> centers(octree.numNodes);
    computeLeafMassCenter<T, T, T>(coordinates.x(), coordinates.y(), coordinates.z(), masses, toInternal, layout.data(),
                                   centers.data());
    upsweep(octree.levelRange, octree.childOffsets, centers.data(), CombineSourceCenter<T>{});
    setMac<T, KeyType>(octree.prefixes, centers, 1.0 / theta, box);

    std::vector<MultipoleType> multipoles(octree.numNodes);
    computeLeafMultipoles(x, y, z, masses.data(), toInternal, layout.data(), centers.data(), multipoles.data());
    upsweepMultipoles(octree.levelRange, octree.childOffsets.data(), centers.data(), multipoles.data());
    for (size_t i = 0; i < multipoles.size(); ++i)
    {
        multipoles[i] = ryoanji::normalize(multipoles[i]);
    }

    T totalMass = std::accumulate(masses.begin(), masses.end(), 0.0);
    EXPECT_TRUE(std::abs(totalMass - multipoles[0][ryoanji::Cqi::mass]) < 1e-6);

    std::vector<T> ax(numParticles, 0);
    std::vector<T> ay(numParticles, 0);
    std::vector<T> az(numParticles, 0);

    auto   t0       = std::chrono::high_resolution_clock::now();
    double egravTot = computeGravity(octree.childOffsets.data(), octree.internalToLeaf.data(), octree.numLeafNodes,
                                     centers.data(), multipoles.data(), layout.data(), 0, octree.numLeafNodes, x, y, z,
                                     h.data(), masses.data(), G, ax.data(), ay.data(), az.data());
    auto   t1       = std::chrono::high_resolution_clock::now();
    double elapsed  = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "Time elapsed for " << numParticles << " particles: " << elapsed << " s, "
              << double(numParticles) / 1e6 / elapsed << " million particles/second" << std::endl;

    // direct sum reference
    std::vector<T> Ax(numParticles, 0);
    std::vector<T> Ay(numParticles, 0);
    std::vector<T> Az(numParticles, 0);
    std::vector<T> potentialReference(numParticles, 0);

    t0 = std::chrono::high_resolution_clock::now();
    directSum(x, y, z, h.data(), masses.data(), numParticles, G, Ax.data(), Ay.data(), Az.data(),
              potentialReference.data());
    t1      = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration<double>(t1 - t0).count();
    
    // relative errors
    std::vector<T> delta(numParticles);
    for (LocalIndex i = 0; i < numParticles; ++i)
    {
        T dx = ax[i] - Ax[i];
        T dy = ay[i] - Ay[i];
        T dz = az[i] - Az[i];

        delta[i] = std::sqrt((dx * dx + dy * dy + dz * dz) / (Ax[i] * Ax[i] + Ay[i] * Ay[i] + Az[i] * Az[i]));
    }

    std::cout << "Time elapsed for direct sum: " << elapsed << " s, " << double(numParticles) / 1e6 / elapsed
              << " million particles/second" << std::endl;

    double refPotSum = 0;
    for (LocalIndex i = 0; i < numParticles; ++i)
    {
        refPotSum += potentialReference[i];
    }
    refPotSum *= 0.5;
    if (refPotSum != 0.0)
        EXPECT_NEAR(std::abs(refPotSum - egravTot) / refPotSum, 0, 1e-2);
    else
        EXPECT_NEAR(std::abs(refPotSum - egravTot), 0, 1e-2);


    std::vector<T> thetas;
//  thetas.push_back(0.2);
//  thetas.push_back(0.4);
//  thetas.push_back(0.6);
//  thetas.push_back(0.8);
    thetas.push_back(1.0);
    for (size_t theta_idx=0; theta_idx < thetas.size(); theta_idx++)
    {
        theta = thetas[theta_idx];
        setMac<T, KeyType>(octree.prefixes, centers, 1.0 / theta, box);

        for (LocalIndex i = 0; i < numParticles; ++i)
        {
            ax[i] = ay[i] = az[i] = 0.0;
        }

        auto   t0       = std::chrono::high_resolution_clock::now();
        double egravTotPBC = computeGravityEwald(octree.childOffsets.data(), octree.internalToLeaf.data(), octree.numLeafNodes,
                                         centers.data(), multipoles.data(), layout.data(), 0, octree.numLeafNodes, x, y, z,
                                         h.data(), masses.data(), G, ax.data(), ay.data(), az.data(), 
                                         box);
        auto   t1       = std::chrono::high_resolution_clock::now();
        double elapsed  = std::chrono::duration<double>(t1 - t0).count();

        cstone::Vec3<T> atot{0.0, 0.0, 0.0};
        for (LocalIndex i = 0; i < numParticles; ++i)
        {
            atot[0] += ax[i];
            atot[1] += ay[i];
            atot[2] += az[i];
            //printf("%23.15f  %23.15f  %23.15f\n", ax[i], ay[i], az[i]);
        }
        T amag = std::sqrt(atot[0]*atot[0] + atot[1]*atot[1] + atot[2]*atot[2]);
        //std::cout << numShells << "/" << maxNumShells << "] Ref. Pot: "  << refPotSum << "    Est. Pot: " << egravTot << "    Est. Pot. PBC: " << egravTotPBC << "    PBC Time: " << elapsed << "s" << std::endl;
        //std::cout << numShells << "/" << maxNumShells << "] Ref. Pot: "  << refPotSum << "    Est. Pot: " << egravTot << "    Est. Pot. PBC: " << egravTotPBC << "Est. Tot a: " << amag << "    PBC Time: " << elapsed << "s" << std::endl;
        std::cout << "theta: "       << theta
            << "    Ref. Pot: "      << refPotSum 
            << "    Est. Pot: "      << egravTot 
            << "    Est. Pot. PBC: " << egravTotPBC 
            << "    Est. Tot |a|: "  << amag 
            << "    PBC Time: "      << elapsed << "s" 
            << std::endl;
    }

}
