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

const int TEST_RNG_SEED = 42;

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

    RandomCoordinates<T, SfcKind<KeyType>> coordinates(numParticles, box, TEST_RNG_SEED);

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
            double egravTotPBC = computeGravityReplica(octree.childOffsets.data(), octree.internalToLeaf.data(), octree.numLeafNodes,
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

template<class T, class KeyType, class MultipoleType>
util::tuple
<
    RandomCoordinates<T, SfcKind<KeyType>>, // coordinates
    std::vector<LocalIndex>,                // layout 
    OctreeData<KeyType, CpuTag>,            // octree
    std::vector<MultipoleType>,             // multipoles
    std::vector<SourceCenterType<T>>,       // centers
    std::vector<T>,                         // masses
    std::vector<T>                          // h
>
makeTestTree(cstone::Box<T> box, LocalIndex numParticles, float theta = 0.6, unsigned bucketSize = 64)
{
    RandomCoordinates<T, SfcKind<KeyType>> coordinates(numParticles, box, TEST_RNG_SEED);

    const T* x = coordinates.x().data();
    const T* y = coordinates.y().data();
    const T* z = coordinates.z().data();

    srand48(TEST_RNG_SEED);

    std::vector<T> masses(numParticles, 1.0/numParticles);
    //std::generate(begin(masses), end(masses), drand48);

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

    std::vector<MultipoleType> multipoles(octree.numNodes);
    computeLeafMultipoles(x, y, z, masses.data(), toInternal, layout.data(), centers.data(), multipoles.data());
    upsweepMultipoles(octree.levelRange, octree.childOffsets.data(), centers.data(), multipoles.data());
    for (size_t i = 0; i < multipoles.size(); ++i)
    {
        multipoles[i] = ryoanji::normalize(multipoles[i]);
    }

    T totalMass = std::accumulate(masses.begin(), masses.end(), 0.0);
    EXPECT_TRUE(std::abs(totalMass - multipoles[0][ryoanji::Cqi::mass]) < 1e-6);

    setMac<T, KeyType>(octree.prefixes, centers, 1.0 / theta, box);

    std::vector<T> h(numParticles, 0.01);

    return {coordinates, layout, octree, multipoles, centers, masses, h};
}

TEST(Gravity, EwaldBasicTests)
{
    EXPECT_EQ(EWALD_GRAVITY_SWITCH::GRAV_ALL,              0);
    EXPECT_EQ(EWALD_GRAVITY_SWITCH::GRAV_NO_CENTRAL_BOX,   1);
    EXPECT_EQ(EWALD_GRAVITY_SWITCH::GRAV_NO_REPLICAS,      2);
    EXPECT_EQ(EWALD_GRAVITY_SWITCH::GRAV_NO_EWALD,         4);
    EXPECT_EQ(EWALD_GRAVITY_SWITCH::GRAV_NO_EWALD
            | EWALD_GRAVITY_SWITCH::GRAV_NO_REPLICAS,      6);

    EXPECT_EQ(   EWALD_GRAVITY_SWITCH::GRAV_NO_EWALD
               + EWALD_GRAVITY_SWITCH::GRAV_NO_REPLICAS
            ,    EWALD_GRAVITY_SWITCH::GRAV_NO_EWALD
               | EWALD_GRAVITY_SWITCH::GRAV_NO_REPLICAS);
}

TEST(Gravity, EwaldBaseline)
{
    //GTEST_SKIP() << "Skipping TreeWalkEwald";

    using T             = double;
    using KeyType       = uint64_t;
    using MultipoleType = ryoanji::CartesianQuadrupole<T>;

    float      theta        = 1.0;
    float      G            = 1.0;
    LocalIndex numParticles = 10000;

    cstone::Box<T> box(-1, 1, cstone::BoundaryType::periodic);

    auto [coordinates, layout, octree, multipoles, centers, masses, h] = makeTestTree<T, KeyType, MultipoleType>(box, numParticles, theta);

    const T* x = coordinates.x().data();
    const T* y = coordinates.y().data();
    const T* z = coordinates.z().data();

    // direct sum reference
    std::vector<T> ax_ref(numParticles, 0);
    std::vector<T> ay_ref(numParticles, 0);
    std::vector<T> az_ref(numParticles, 0);
    std::vector<T>  u_ref(numParticles, 0);

    directSum(x, y, z, h.data(), masses.data(), numParticles, G, ax_ref.data(), ay_ref.data(), az_ref.data(), u_ref.data());

    std::vector<T> ax(numParticles, 0);
    std::vector<T> ay(numParticles, 0);
    std::vector<T> az(numParticles, 0);

    double utot = computeGravity(octree.childOffsets.data(), octree.internalToLeaf.data(), octree.numLeafNodes,
                                 centers.data(), multipoles.data(), layout.data(), 0, octree.numLeafNodes, x, y, z,
                                 h.data(), masses.data(), G, ax.data(), ay.data(), az.data());
    
    // relative errors
//  std::vector<T> delta(numParticles);
//  for (LocalIndex i = 0; i < numParticles; ++i)
//  {
//      T dx = ax[i] - Ax[i];
//      T dy = ay[i] - Ay[i];
//      T dz = az[i] - Az[i];

//      delta[i] = std::sqrt((dx * dx + dy * dy + dz * dz) / (Ax[i] * Ax[i] + Ay[i] * Ay[i] + Az[i] * Az[i]));
//  }

    double utot_ref = 0;
    for (LocalIndex i = 0; i < numParticles; ++i)
    {
        utot_ref += u_ref[i];
    }
    utot_ref *= 0.5;
    if (utot_ref != 0.0)
        EXPECT_NEAR(std::abs(utot_ref - utot) / utot_ref, 0, 1e-2);
    else
        EXPECT_NEAR(std::abs(utot_ref - utot), 0, 1e-2);

}

TEST(Gravity, EwaldNoPBCContribution)
{
    //GTEST_SKIP() << "Skipping TreeWalkEwald";

    using T             = double;
    using KeyType       = uint64_t;
    using MultipoleType = ryoanji::CartesianQuadrupole<T>;

    cstone::Box<T> box(-1, 1, cstone::BoundaryType::periodic);

    float      theta        = 1.0;
    float      G            = 1.0;
    LocalIndex numParticles = 10000;

    auto [coordinates, layout, octree, multipoles, centers, masses, h] = makeTestTree<T, KeyType, MultipoleType>(box, numParticles, theta);

    const T* x = coordinates.x().data();
    const T* y = coordinates.y().data();
    const T* z = coordinates.z().data();

    std::vector<T> ax(numParticles, 0);
    std::vector<T> ay(numParticles, 0);
    std::vector<T> az(numParticles, 0);
    
    // direct sum reference
    std::vector<T> ax_ref(numParticles, 0);
    std::vector<T> ay_ref(numParticles, 0);
    std::vector<T> az_ref(numParticles, 0);

    double utot_ref = computeGravity(octree.childOffsets.data(), octree.internalToLeaf.data(), octree.numLeafNodes,
                                     centers.data(), multipoles.data(), layout.data(), 0, octree.numLeafNodes, x, y, z,
                                     h.data(), masses.data(), G, ax_ref.data(), ay_ref.data(), az_ref.data());
    cstone::Vec3<T> atot_ref{0.0, 0.0, 0.0};
    for (LocalIndex i = 0; i < numParticles; ++i)
    {
        atot_ref[0] += ax_ref[i];
        atot_ref[1] += ay_ref[i];
        atot_ref[2] += az_ref[i];
    }
    T amag_ref = std::sqrt(atot_ref[0]*atot_ref[0] + atot_ref[1]*atot_ref[1] + atot_ref[2]*atot_ref[2]);

    printf("# -- EWALD GRAVITY NO PBC  ----------------------------------------------------\n");

    {
        std::fill_n(ax.begin(), numParticles, 0);
        std::fill_n(ay.begin(), numParticles, 0);
        std::fill_n(az.begin(), numParticles, 0);

        double utot = computeGravityEwald(octree.childOffsets.data(), octree.internalToLeaf.data(), octree.numLeafNodes,
                                     centers.data(), multipoles.data(), layout.data(), 0, octree.numLeafNodes, x, y, z,
                                     h.data(), masses.data(), G, ax.data(), ay.data(), az.data(), 
                                     box, 0, 0, 0, GRAV_NO_REPLICAS | GRAV_NO_EWALD);

        cstone::Vec3<T> atot{0.0, 0.0, 0.0};
        for (LocalIndex i = 0; i < numParticles; ++i)
        {
            atot[0] += ax[i];
            atot[1] += ay[i];
            atot[2] += az[i];
        }
        T amag = std::sqrt(atot[0]*atot[0] + atot[1]*atot[1] + atot[2]*atot[2]);

        printf("numShells: %3i"
             "  Ref. Utot: %23.15e"
             "  Ref. |a|: %23.15e"
             "  Ewald Utot: %23.15e"
             "  Ewald |a|: %23.15e"
             "\n",
             0, utot_ref, amag_ref, utot, amag);
    }
}

TEST(Gravity, Ewald)
{
    GTEST_SKIP() << "Skipping Ewald";

    using T             = double;
    using KeyType       = uint64_t;
    using MultipoleType = ryoanji::CartesianQuadrupole<T>;

    cstone::Box<T> box(-1, 1, cstone::BoundaryType::periodic);

    float      theta        = 1.0;
    float      G            = 1.0;
    LocalIndex numParticles = 10000;

    auto [coordinates, layout, octree, multipoles, centers, masses, h] = makeTestTree<T, KeyType, MultipoleType>(box, numParticles, theta);

    const T* x = coordinates.x().data();
    const T* y = coordinates.y().data();
    const T* z = coordinates.z().data();

    std::vector<T> ax(numParticles, 0);
    std::vector<T> ay(numParticles, 0);
    std::vector<T> az(numParticles, 0);
    
    // direct sum reference
    std::vector<T> ax_ref(numParticles, 0);
    std::vector<T> ay_ref(numParticles, 0);
    std::vector<T> az_ref(numParticles, 0);

    double utot_ref = computeGravityEwald(octree.childOffsets.data(), octree.internalToLeaf.data(), octree.numLeafNodes,
                                     centers.data(), multipoles.data(), layout.data(), 0, octree.numLeafNodes, x, y, z,
                                     h.data(), masses.data(), G, ax_ref.data(), ay_ref.data(), az_ref.data(), 
                                     box, 1);

    cstone::Vec3<T> atot_ref{0.0, 0.0, 0.0};
    for (LocalIndex i = 0; i < numParticles; ++i)
    {
        atot_ref[0] += ax_ref[i];
        atot_ref[1] += ay_ref[i];
        atot_ref[2] += az_ref[i];
    }
    T amag_ref = std::sqrt(atot_ref[0]*atot_ref[0] + atot_ref[1]*atot_ref[1] + atot_ref[2]*atot_ref[2]);

    for (auto numShells = 0; numShells <= 8; numShells++)
    {
        std::fill_n(ax.begin(), numParticles, 0);
        std::fill_n(ay.begin(), numParticles, 0);
        std::fill_n(az.begin(), numParticles, 0);

        double hCut      = 1.8 + numShells;
        double ewaldCut  = 1.6 + numShells;
        double utot = computeGravityEwald(octree.childOffsets.data(), octree.internalToLeaf.data(), octree.numLeafNodes,
                                     centers.data(), multipoles.data(), layout.data(), 0, octree.numLeafNodes, x, y, z,
                                     h.data(), masses.data(), G, ax.data(), ay.data(), az.data(), 
                                     box, numShells, hCut, ewaldCut, GRAV_ALL);

        cstone::Vec3<T> atot{0.0, 0.0, 0.0};
        for (LocalIndex i = 0; i < numParticles; ++i)
        {
            atot[0] += ax[i];
            atot[1] += ay[i];
            atot[2] += az[i];
        }
        T amag = std::sqrt(atot[0]*atot[0] + atot[1]*atot[1] + atot[2]*atot[2]);

        printf("numShells: %3i"
             "  Ref. Utot: %23.15e"
             "  Ref. |a|: %23.15e"
             "  Ewald Utot: %23.15e"
             "  Ewald |a|: %23.15e"
             "\n",
             numShells, utot_ref, amag_ref, utot, amag);

//      if (utot_ref != 0.0)
//          EXPECT_NEAR(std::abs(utot_ref - utot) / utot_ref, 0, 1e-2);
//      else
//          EXPECT_NEAR(std::abs(utot_ref - utot), 0, 1e-2);
    }

    // relative errors
//  std::vector<T> delta(numParticles);
//  for (LocalIndex i = 0; i < numParticles; ++i)
//  {
//      T dx = ax[i] - Ax[i];
//      T dy = ay[i] - Ay[i];
//      T dz = az[i] - Az[i];

//      delta[i] = std::sqrt((dx * dx + dy * dy + dz * dz) / (Ax[i] * Ax[i] + Ay[i] * Ay[i] + Az[i] * Az[i]));
//  }

#if 0

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
#endif
}

template <class T>
void runEwaldPartialsTest(const float theta, const LocalIndex numParticles, const int maxShells, const int maxEwaldShells, cstone::Box<T> box, const float G = 1.0)
{
    using KeyType       = uint64_t;
    using MultipoleType = ryoanji::CartesianQuadrupole<T>;

    //cstone::Box<T> box(-1, 1, cstone::BoundaryType::periodic);

    printf("# -- EwaldPartials Parameters -----\n");
    printf("#  theta             :   %8.4f     \n", theta);
    printf("#  G                 :   %8.4f     \n", G);
    printf("#  numParticles      :   %8i       \n", numParticles);
    printf("#  maxShells         :   %8i       \n", maxShells);
    printf("#  maxEwaldShells    :   %8i       \n", maxEwaldShells);
    printf("#  box               : w/h/d  %g  %g  %g\n", box.lx(), box.ly(), box.lz());
    printf("#                    : [%g  %g]  ", box.xmin(), box.xmax());
    printf(                       "[%g  %g]  ", box.ymin(), box.ymax());
    printf(                       "[%g  %g]\n", box.zmin(), box.zmax());
    printf("# ---------------------------------\n");

    auto [coordinates, layout, octree, multipoles, centers, masses, h] = makeTestTree<T, KeyType, MultipoleType>(box, numParticles, theta);

    const T* x = coordinates.x().data();
    const T* y = coordinates.y().data();
    const T* z = coordinates.z().data();

    std::vector<T> ax(numParticles, 0);
    std::vector<T> ay(numParticles, 0);
    std::vector<T> az(numParticles, 0);
    
    // direct sum reference
    std::vector<T> ax_ref(numParticles, 0);
    std::vector<T> ay_ref(numParticles, 0);
    std::vector<T> az_ref(numParticles, 0);

    double utot_ref = computeGravityEwald(octree.childOffsets.data(), octree.internalToLeaf.data(), octree.numLeafNodes,
                                     centers.data(), multipoles.data(), layout.data(), 0, octree.numLeafNodes, 
                                     x, y, z, h.data(), masses.data(), G, ax_ref.data(), ay_ref.data(), az_ref.data(), 
                                     box, 1);

    T axtot_ref = std::accumulate(ax_ref.begin(), ax_ref.end(), 0.0);
    T aytot_ref = std::accumulate(ay_ref.begin(), ay_ref.end(), 0.0);
    T aztot_ref = std::accumulate(az_ref.begin(), az_ref.end(), 0.0);
    T amag_ref  = std::sqrt(axtot_ref*axtot_ref + aytot_ref*aytot_ref + aztot_ref*aztot_ref);

    printf("# %8s %23s %23s %23s %23s %23s %23s %23s\n",
            "numShells",
            "Ref. Utot",
            "Ref. |a|",
            "Ewald Utot",
            "Ewald |a|",
            "Ewald ax",
            "Ewald ay",
            "Ewald az");
    printf("# %8s %23s %23s %23s %23s %23s %23s %23s\n",
            std::string( 8,'-').c_str(),
            std::string(23,'-').c_str(),
            std::string(23,'-').c_str(),
            std::string(23,'-').c_str(),
            std::string(23,'-').c_str(),
            std::string(23,'-').c_str(),
            std::string(23,'-').c_str(),
            std::string(23,'-').c_str());

    printf("# -- FULL PBC GRAVITY --\n");
    for (auto numShells = 0; numShells <= maxShells; numShells++)
    for (auto numEwaldShells = 0; numEwaldShells <= maxEwaldShells; numEwaldShells++)
    {
        std::fill(ax.begin(), ax.end(), 0);
        std::fill(ay.begin(), ay.end(), 0);
        std::fill(az.begin(), az.end(), 0);

        double hCut        = 0.8 + numEwaldShells + numShells;
        double ewaldCut    = 0.6 + numEwaldShells + numShells;
        double alpha_scale = 2.0;
        double utot = computeGravityEwald(octree.childOffsets.data(), octree.internalToLeaf.data(), octree.numLeafNodes,
                                     centers.data(), multipoles.data(), layout.data(), 0, octree.numLeafNodes, x, y, z,
                                     h.data(), masses.data(), G, ax.data(), ay.data(), az.data(), 
                                     box, numShells, hCut, ewaldCut, alpha_scale, GRAV_ALL);

        T axtot = std::accumulate(ax.begin(), ax.end(), 0.0);
        T aytot = std::accumulate(ay.begin(), ay.end(), 0.0);
        T aztot = std::accumulate(az.begin(), az.end(), 0.0);
        T amag  = std::sqrt(axtot*axtot + aytot*aytot + aztot*aztot);

        printf("%3i %3i %23.15e %23.15e %23.15e %23.15e %23.15e %23.15e %23.15e\n",
             numShells, numEwaldShells, utot_ref, amag_ref, utot, amag, axtot, aytot, aztot);
    }

    printf("# -- NO EWALD --\n");
    for (auto numShells = 0; numShells <= maxShells; numShells++)
    {
        std::fill(ax.begin(), ax.end(), 0);
        std::fill(ay.begin(), ay.end(), 0);
        std::fill(az.begin(), az.end(), 0);

        double hCut        = 0.0;
        double ewaldCut    = 0.0;
        double alpha_scale = 0.0;
        double utot = computeGravityEwald(octree.childOffsets.data(), octree.internalToLeaf.data(), octree.numLeafNodes,
                                     centers.data(), multipoles.data(), layout.data(), 0, octree.numLeafNodes, x, y, z,
                                     h.data(), masses.data(), G, ax.data(), ay.data(), az.data(), 
                                     box, numShells, hCut, ewaldCut, alpha_scale, GRAV_NO_EWALD | GRAV_NO_CENTRAL_BOX);

        T axtot = std::accumulate(ax.begin(), ax.end(), 0.0);
        T aytot = std::accumulate(ay.begin(), ay.end(), 0.0);
        T aztot = std::accumulate(az.begin(), az.end(), 0.0);
        T amag  = std::sqrt(axtot*axtot + aytot*aytot + aztot*aztot);

        printf("%3i %3i %23.15e %23.15e %23.15e %23.15e %23.15e %23.15e %23.15e\n",
             numShells, 0, utot_ref, amag_ref, utot, amag, axtot, aytot, aztot);
//      for (auto i=LocalIndex(0); i < numParticles; i++)
//      {
//          printf("%3i %3i  %23.15e %23.15e %23.15e\n",
//               numShells, i, ax.data()[i], ay.data()[i], az.data()[i]);
//      }
    }

    printf("# -- ONLY EWALD --\n");

    for (auto numShells = 0; numShells <= maxShells; numShells++)
    for (auto numEwaldShells = 0; numEwaldShells <= maxEwaldShells; numEwaldShells++)
    {
        std::fill(ax.begin(), ax.end(), 0);
        std::fill(ay.begin(), ay.end(), 0);
        std::fill(az.begin(), az.end(), 0);

        double hCut      = 0.8 + numEwaldShells + numShells;
        double ewaldCut  = 0.6 + numEwaldShells + numShells;
        double alpha_scale = 2.0;
        double utot = computeGravityEwald(octree.childOffsets.data(), octree.internalToLeaf.data(), octree.numLeafNodes,
                                     centers.data(), multipoles.data(), layout.data(), 0, octree.numLeafNodes, x, y, z,
                                     h.data(), masses.data(), G, ax.data(), ay.data(), az.data(), 
                                     box, numShells, hCut, ewaldCut, alpha_scale,
                                     EWALD_GRAVITY_SWITCH::GRAV_NO_CENTRAL_BOX | EWALD_GRAVITY_SWITCH::GRAV_NO_REPLICAS);

        T axtot = std::accumulate(ax.begin(), ax.end(), 0.0);
        T aytot = std::accumulate(ay.begin(), ay.end(), 0.0);
        T aztot = std::accumulate(az.begin(), az.end(), 0.0);
        T amag  = std::sqrt(axtot*axtot + aytot*aytot + aztot*aztot);

        printf("%3i %3i %23.15e %23.15e %23.15e %23.15e %23.15e %23.15e %23.15e\n",
             numShells, numEwaldShells, utot_ref, amag_ref, utot, amag, axtot, aytot, aztot);
    }

    return;

    printf("# -- ONLY EWALD REALSPACE --\n");

    for (auto numShells = 0; numShells <= maxShells; numShells++)
    for (auto numEwaldShells = 0; numEwaldShells <= maxEwaldShells; numEwaldShells++)
    {
        std::fill(ax.begin(), ax.end(), 0);
        std::fill(ay.begin(), ay.end(), 0);
        std::fill(az.begin(), az.end(), 0);

        double hCut      = 0.8 + numEwaldShells + numShells;
        double ewaldCut  = 0.6 + numEwaldShells + numShells;
        double alpha_scale = 2.0;
        double utot = computeGravityEwald(octree.childOffsets.data(), octree.internalToLeaf.data(), octree.numLeafNodes,
                                     centers.data(), multipoles.data(), layout.data(), 0, octree.numLeafNodes, x, y, z,
                                     h.data(), masses.data(), G, ax.data(), ay.data(), az.data(), 
                                     box, numShells, hCut, ewaldCut, alpha_scale,
                                     GRAV_NO_CENTRAL_BOX 
                                     | GRAV_NO_REPLICAS
                                     | GRAV_NO_EWALD_KSPACE);

        T axtot = std::accumulate(ax.begin(), ax.end(), 0.0);
        T aytot = std::accumulate(ay.begin(), ay.end(), 0.0);
        T aztot = std::accumulate(az.begin(), az.end(), 0.0);
        T amag  = std::sqrt(axtot*axtot + aytot*aytot + aztot*aztot);

        printf("%3i %3i %23.15e %23.15e %23.15e %23.15e %23.15e %23.15e %23.15e\n",
             numShells, numEwaldShells, utot_ref, amag_ref, utot, amag, axtot, aytot, aztot);
    }

    printf("# -- ONLY EWALD KSPACE --\n");

    for (auto numShells = 0; numShells <= maxShells; numShells++)
    for (auto numEwaldShells = 0; numEwaldShells <= maxEwaldShells; numEwaldShells++)
    {
        std::fill(ax.begin(), ax.end(), 0);
        std::fill(ay.begin(), ay.end(), 0);
        std::fill(az.begin(), az.end(), 0);

        double hCut        = 0.8 + numEwaldShells + numShells;
        double ewaldCut    = 0.6 + numEwaldShells + numShells;
        double alpha_scale = 2.0;
        double utot = computeGravityEwald(octree.childOffsets.data(), octree.internalToLeaf.data(), octree.numLeafNodes,
                                     centers.data(), multipoles.data(), layout.data(), 0, octree.numLeafNodes, x, y, z,
                                     h.data(), masses.data(), G, ax.data(), ay.data(), az.data(), 
                                     box, numShells, hCut, ewaldCut, alpha_scale,
                                     GRAV_NO_CENTRAL_BOX 
                                     | GRAV_NO_REPLICAS
                                     | GRAV_NO_EWALD_REALSPACE);

        T axtot = std::accumulate(ax.begin(), ax.end(), 0.0);
        T aytot = std::accumulate(ay.begin(), ay.end(), 0.0);
        T aztot = std::accumulate(az.begin(), az.end(), 0.0);
        T amag  = std::sqrt(axtot*axtot + aytot*aytot + aztot*aztot);

        printf("%3i %3i %23.15e %23.15e %23.15e %23.15e %23.15e %23.15e %23.15e\n",
             numShells, numEwaldShells, utot_ref, amag_ref, utot, amag, axtot, aytot, aztot);
    }
}

template <class T>
void runEwaldAlphaScaleTest(const float theta, const LocalIndex numParticles, const int maxShells, const float minAlphaScale, const float maxAlphaScale, const int numSteps, cstone::Box<T> box, const float G = 1.0)
{
    using KeyType       = uint64_t;
    using MultipoleType = ryoanji::CartesianQuadrupole<T>;

    //cstone::Box<T> box(-1, 1, cstone::BoundaryType::periodic);

    printf("# -- EwaldPartials Parameters -----\n");
    printf("#  theta             :   %8.4f     \n", theta);
    printf("#  G                 :   %8.4f     \n", G);
    printf("#  numParticles      :   %8i       \n", numParticles);
    printf("#  maxShells         :   %8i       \n", maxShells);
    printf("#  AlphaScale        :   %4.2f %4.2f  steps %i\n", minAlphaScale, maxAlphaScale, numSteps);
    printf("#  box               : w/h/d  %g  %g  %g\n", box.lx(), box.ly(), box.lz());
    printf("#                    : [%g  %g]  ", box.xmin(), box.xmax());
    printf(                       "[%g  %g]  ", box.ymin(), box.ymax());
    printf(                       "[%g  %g]\n", box.zmin(), box.zmax());
    printf("# ---------------------------------\n");

    auto [coordinates, layout, octree, multipoles, centers, masses, h] = makeTestTree<T, KeyType, MultipoleType>(box, numParticles, theta);

    const T* x = coordinates.x().data();
    const T* y = coordinates.y().data();
    const T* z = coordinates.z().data();

    std::vector<T> ax(numParticles, 0);
    std::vector<T> ay(numParticles, 0);
    std::vector<T> az(numParticles, 0);
    
    // direct sum reference
    std::vector<T> ax_ref(numParticles, 0);
    std::vector<T> ay_ref(numParticles, 0);
    std::vector<T> az_ref(numParticles, 0);

    double utot_ref = computeGravityEwald(octree.childOffsets.data(), octree.internalToLeaf.data(), octree.numLeafNodes,
                                     centers.data(), multipoles.data(), layout.data(), 0, octree.numLeafNodes, 
                                     x, y, z, h.data(), masses.data(), G, ax_ref.data(), ay_ref.data(), az_ref.data(), 
                                     box, 1);

    T axtot_ref = std::accumulate(ax_ref.begin(), ax_ref.end(), 0.0);
    T aytot_ref = std::accumulate(ay_ref.begin(), ay_ref.end(), 0.0);
    T aztot_ref = std::accumulate(az_ref.begin(), az_ref.end(), 0.0);
    T amag_ref  = std::sqrt(axtot_ref*axtot_ref + aytot_ref*aytot_ref + aztot_ref*aztot_ref);

    printf("# %8s %23s %23s %23s %23s %23s %23s %23s\n",
            "numShells",
            "Ref. Utot",
            "Ref. |a|",
            "Ewald Utot",
            "Ewald |a|",
            "Ewald ax",
            "Ewald ay",
            "Ewald az");
    printf("# %8s %23s %23s %23s %23s %23s %23s %23s\n",
            std::string( 8,'-').c_str(),
            std::string(23,'-').c_str(),
            std::string(23,'-').c_str(),
            std::string(23,'-').c_str(),
            std::string(23,'-').c_str(),
            std::string(23,'-').c_str(),
            std::string(23,'-').c_str(),
            std::string(23,'-').c_str());

    printf("# -- FULL PBC GRAVITY --\n");
    double hCut = 2.8;
    double ewaldCut = 2.6;


    auto steps = (numSteps == 0) ? 0 : std::max(numSteps, 2);

    auto dalpha_scale = (maxAlphaScale - minAlphaScale) / (steps-1);

    auto numShells = 1;

    printf("# -- FULL PBC GRAVITY --\n");
    for (auto i = 0; i < steps; i++)
    {
        std::fill(ax.begin(), ax.end(), 0);
        std::fill(ay.begin(), ay.end(), 0);
        std::fill(az.begin(), az.end(), 0);

        //double alpha_scale = minAlphaScale + std::sin(M_PI/2*double(i)/(steps-1)) * (maxAlphaScale - minAlphaScale);
        double alpha_scale = minAlphaScale + std::pow(2,i) - 1;
        double hCut        = 2.8;
        double ewaldCut    = 2.6;
        int    numShells   = 1;

        double utot = computeGravityEwald(octree.childOffsets.data(), octree.internalToLeaf.data(), octree.numLeafNodes,
                                     centers.data(), multipoles.data(), layout.data(), 0, octree.numLeafNodes, x, y, z,
                                     h.data(), masses.data(), G, ax.data(), ay.data(), az.data(), 
                                     box, numShells, hCut, ewaldCut, alpha_scale, GRAV_ALL);

        T axtot = std::accumulate(ax.begin(), ax.end(), 0.0);
        T aytot = std::accumulate(ay.begin(), ay.end(), 0.0);
        T aztot = std::accumulate(az.begin(), az.end(), 0.0);
        T amag  = std::sqrt(axtot*axtot + aytot*aytot + aztot*aztot);

        printf("%9.3f %23.15e %23.15e %23.15e %23.15e %23.15e %23.15e %23.15e\n",
             alpha_scale, utot_ref, amag_ref, utot, amag, axtot, aytot, aztot);
    }

    printf("# -- FULL PBC GRAVITY v2 --\n");
    for (auto i = 0; i < steps; i++)
    for (auto numEwaldShells = 0; numEwaldShells <= maxShells; numEwaldShells++)
    {
        std::fill(ax.begin(), ax.end(), 0);
        std::fill(ay.begin(), ay.end(), 0);
        std::fill(az.begin(), az.end(), 0);

        //double alpha_scale = minAlphaScale + std::sin(M_PI/2*double(i)/(steps-1)) * (maxAlphaScale - minAlphaScale);
        //double alpha_scale = minAlphaScale + std::pow(2,i) - 1;
        double alpha_scale = minAlphaScale + i * dalpha_scale;
        int    numShells   = 1;
        double hCut        = 0.8 + numShells + numEwaldShells;
        double ewaldCut    = 0.6 + numShells + numEwaldShells;

        double utot = computeGravityEwald(octree.childOffsets.data(), octree.internalToLeaf.data(), octree.numLeafNodes,
                                     centers.data(), multipoles.data(), layout.data(), 0, octree.numLeafNodes, x, y, z,
                                     h.data(), masses.data(), G, ax.data(), ay.data(), az.data(), 
                                     box, numShells, hCut, ewaldCut, alpha_scale, GRAV_ALL);

        T axtot = std::accumulate(ax.begin(), ax.end(), 0.0);
        T aytot = std::accumulate(ay.begin(), ay.end(), 0.0);
        T aztot = std::accumulate(az.begin(), az.end(), 0.0);
        T amag  = std::sqrt(axtot*axtot + aytot*aytot + aztot*aztot);

        printf("%9.3f %3i %23.15e %23.15e %23.15e %23.15e %23.15e %23.15e %23.15e\n",
             alpha_scale, numEwaldShells, utot_ref, amag_ref, utot, amag, axtot, aytot, aztot);
    }

    printf("# -- MODIFIED EWALD GRAVITY --\n");
    for (auto numShells = 0; numShells <= maxShells; numShells++)
    {
        std::fill(ax.begin(), ax.end(), 0);
        std::fill(ay.begin(), ay.end(), 0);
        std::fill(az.begin(), az.end(), 0);

        double hCut        = 0;
        double ewaldCut    = numShells;
        double alpha_scale = 0.0;

        double utot = computeGravityEwald(octree.childOffsets.data(), octree.internalToLeaf.data(), octree.numLeafNodes,
                                     centers.data(), multipoles.data(), layout.data(), 0, octree.numLeafNodes, x, y, z,
                                     h.data(), masses.data(), G, ax.data(), ay.data(), az.data(), 
                                     box, 1, hCut, ewaldCut, alpha_scale); //, GRAV_NO_REPLICAS | GRAV_NO_CENTRAL_BOX);

        T axtot = std::accumulate(ax.begin(), ax.end(), 0.0);
        T aytot = std::accumulate(ay.begin(), ay.end(), 0.0);
        T aztot = std::accumulate(az.begin(), az.end(), 0.0);
        T amag  = std::sqrt(axtot*axtot + aytot*aytot + aztot*aztot);

        printf("%9.3f %5.3f %23.15e %23.15e %23.15e %23.15e %23.15e %23.15e %23.15e\n",
             alpha_scale, ewaldCut, utot_ref, amag_ref, utot, amag, axtot, aytot, aztot);
    }

    printf("# -- NO EWALD --\n");
    for (auto numShells = 0; numShells <= maxShells; numShells++)
    {
        std::fill(ax.begin(), ax.end(), 0);
        std::fill(ay.begin(), ay.end(), 0);
        std::fill(az.begin(), az.end(), 0);

        double hCut        = 0;
        double ewaldCut    = 0;
        double alpha_scale = 0.0;
        double utot = computeGravityEwald(octree.childOffsets.data(), octree.internalToLeaf.data(), octree.numLeafNodes,
                                     centers.data(), multipoles.data(), layout.data(), 0, octree.numLeafNodes, x, y, z,
                                     h.data(), masses.data(), G, ax.data(), ay.data(), az.data(), 
                                     box, numShells, hCut, ewaldCut, alpha_scale, GRAV_NO_EWALD); // | GRAV_NO_CENTRAL_BOX);

        T axtot = std::accumulate(ax.begin(), ax.end(), 0.0);
        T aytot = std::accumulate(ay.begin(), ay.end(), 0.0);
        T aztot = std::accumulate(az.begin(), az.end(), 0.0);
        T amag  = std::sqrt(axtot*axtot + aytot*aytot + aztot*aztot);

        printf("%3i %23.15e %23.15e %23.15e %23.15e %23.15e %23.15e %23.15e\n",
             numShells, utot_ref, amag_ref, utot, amag, axtot, aytot, aztot);
    }
}

TEST(Gravity, EwaldPartials)
{
    //GTEST_SKIP() << "Skipping EwaldPartials";

    using T = double;

//  runEwaldPartialsTest(0.0, 1,   8, 2, cstone::Box<T>(-1, 1, cstone::BoundaryType::periodic));
    //runEwaldPartialsTest(0.0, 2,   8, 2, cstone::Box<T>(-1, 1, cstone::BoundaryType::periodic));
    //runEwaldPartialsTest(0.0, 2,   8, cstone::Box<T>(-1, 1, cstone::BoundaryType::periodic));
    //runEwaldPartialsTest(0.0, 3,   8, 2, cstone::Box<T>(-1, 1, cstone::BoundaryType::periodic));
    //runEwaldPartialsTest(0.0, 3,   1, 1, cstone::Box<T>(-1, 1, cstone::BoundaryType::periodic));
    //runEwaldPartialsTest(0.0, 100, 8, 9, cstone::Box<T>(-1, 1, cstone::BoundaryType::periodic));

//  runEwaldPartialsTest(1.0, 1, 8, cstone::Box<T>(-1, 1, cstone::BoundaryType::periodic));
//  runEwaldPartialsTest(1.0, 2, 8, cstone::Box<T>(-1, 1, cstone::BoundaryType::periodic));
//  runEwaldPartialsTest(1.0, 3, 8, cstone::Box<T>(-1, 1, cstone::BoundaryType::periodic));

//  runEwaldPartialsTest(1.0, 10000, 8, cstone::Box<T>(-1, 1, cstone::BoundaryType::periodic));
    //runEwaldPartialsTest(0.6, 10000, 8, 2, cstone::Box<T>(-1, 1, cstone::BoundaryType::periodic));

    runEwaldAlphaScaleTest(1.0, 1,   8, 0.0, 2.0, 3, cstone::Box<T>(-1, 1, cstone::BoundaryType::periodic));
    runEwaldAlphaScaleTest(1.0, 2,   8, 0.0, 2.0, 3, cstone::Box<T>(-1, 1, cstone::BoundaryType::periodic));
    //runEwaldAlphaScaleTest(0.0, 3,   8, 0.0, 2.0, 3, cstone::Box<T>(-1, 1, cstone::BoundaryType::periodic));
    runEwaldAlphaScaleTest(0.0, 100,   3, 0.0, 2.0, 3, cstone::Box<T>(-1, 1, cstone::BoundaryType::periodic));
    runEwaldAlphaScaleTest(1.0, 100,   8, 0.0, 2.0, 3, cstone::Box<T>(-1, 1, cstone::BoundaryType::periodic));

    runEwaldAlphaScaleTest(1.0, 100,   8, 0.0, 16.0, 17, cstone::Box<T>(-1, 1, cstone::BoundaryType::periodic));
    runEwaldAlphaScaleTest(0.0, 100,   8, 0.0, 16.0, 17, cstone::Box<T>(-1, 1, cstone::BoundaryType::periodic));

    //runEwaldAlphaScaleTest(1.0, 10000,   8, 0.0, 2.0, 3, cstone::Box<T>(-1, 1, cstone::BoundaryType::periodic));
    //runEwaldAlphaScaleTest(0.6, 10000,   8, 0.0, 2.0, 3, cstone::Box<T>(-1, 1, cstone::BoundaryType::periodic));
}
