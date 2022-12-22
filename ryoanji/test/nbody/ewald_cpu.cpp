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

#include "cstone/util/tuple.hpp"
#include "cstone/sfc/box.hpp"
#include "coord_samples/random.hpp"
#include "ryoanji/nbody/traversal_cpu.hpp"
#include "ryoanji/nbody/traversal_ewald_cpu.hpp"
#include "ryoanji/nbody/upsweep_cpu.hpp"

using namespace cstone;
using namespace ryoanji;

const int TEST_RNG_SEED = 42;

const int verbose = 1;

#define V(level) if ((level) == verbose)

#if 0
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
#endif

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
makeTestTree(cstone::Box<T> box, LocalIndex numParticles, float theta = 0.6, bool random_masses = true, unsigned bucketSize = 64)
{
    RandomCoordinates<T, SfcKind<KeyType>> coordinates(numParticles, box, TEST_RNG_SEED);

    const T* x = coordinates.x().data();
    const T* y = coordinates.y().data();
    const T* z = coordinates.z().data();

    srand48(TEST_RNG_SEED);

    std::vector<T> masses(numParticles, 1.0/numParticles);
    if (random_masses) std::generate(begin(masses), end(masses), drand48);

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
//  EXPECT_EQ(EWALD_GRAVITY_SWITCH::GRAV_ALL,                1 << 0);
//  EXPECT_EQ(EWALD_GRAVITY_SWITCH::GRAV_NO_CENTRAL_BOX,     1 << 1);
//  EXPECT_EQ(EWALD_GRAVITY_SWITCH::GRAV_NO_REPLICAS,        1 << 2);
//  EXPECT_EQ(EWALD_GRAVITY_SWITCH::GRAV_NO_EWALD,           1 << 3);
//  EXPECT_EQ(EWALD_GRAVITY_SWITCH::GRAV_NO_EWALD_REALSPACE, 1 << 4);
//  EXPECT_EQ(EWALD_GRAVITY_SWITCH::GRAV_NO_EWALD_KSPACE,    1 << 5);

    EXPECT_EQ(   EWALD_GRAVITY_SWITCH::GRAV_NO_EWALD
               + EWALD_GRAVITY_SWITCH::GRAV_NO_REPLICAS
            ,    EWALD_GRAVITY_SWITCH::GRAV_NO_EWALD
               | EWALD_GRAVITY_SWITCH::GRAV_NO_REPLICAS);
}

TEST(Gravity, EwaldBaseline)
{
    using T             = double;
    using KeyType       = uint64_t;
    using MultipoleType = ryoanji::CartesianQuadrupole<T>;

    auto test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();

    float      G            = 1.0;
    LocalIndex numParticles = 100;
    std::vector<T> thetas   = {0.0, 0.5, 1.0};

    cstone::Box<T> box(-1, 1, cstone::BoundaryType::periodic);

    V(1) printf("# %28s | %6s %5s %21s\n", "Test", "nPart", "Theta", "min/50/10/1/max Error");
    V(1) printf("# %28s | %6s %5s %21s\n", "----", "-----", "-----", "---------------------");

    for (auto theta : thetas)
    {
        auto [coordinates, layout, octree, multipoles, centers, masses, h] = makeTestTree<T, KeyType, MultipoleType>(box, numParticles, theta);

        const T* x = coordinates.x().data();
        const T* y = coordinates.y().data();
        const T* z = coordinates.z().data();

        std::vector<T> ax_ref(numParticles, 0);
        std::vector<T> ay_ref(numParticles, 0);
        std::vector<T> az_ref(numParticles, 0);
        std::vector<T>  u_ref(numParticles, 0);

        double utot_ref = computeGravity(octree.childOffsets.data(), octree.internalToLeaf.data(), octree.numLeafNodes,
                                     centers.data(), multipoles.data(), layout.data(), 0, octree.numLeafNodes, x, y, z,
                                     h.data(), masses.data(), G, ax_ref.data(), ay_ref.data(), az_ref.data());

        std::vector<T> ax(numParticles, 0);
        std::vector<T> ay(numParticles, 0);
        std::vector<T> az(numParticles, 0);

        double hCut             = 0.0;
        double ewaldCut         = 0.0;
        double alpha_scale      = 0.0;
        int    numReplicaShells = 0;

        double utot = computeGravityEwald(octree.childOffsets.data(), octree.internalToLeaf.data(), octree.numLeafNodes,
                                     centers.data(), multipoles.data(), layout.data(), 0, octree.numLeafNodes, x, y, z,
                                     h.data(), masses.data(), G, ax.data(), ay.data(), az.data(), 
                                     box, numReplicaShells, hCut, ewaldCut, alpha_scale); //, GRAV_NO_EWALD | GRAV_NO_REPLICAS);
    
        // relative errors
        std::vector<T> delta(numParticles);
        for (LocalIndex i = 0; i < numParticles; ++i)
        {
            T dx = ax[i] - ax_ref[i];
            T dy = ay[i] - ay_ref[i];
            T dz = az[i] - az_ref[i];

            delta[i] = std::sqrt((dx * dx + dy * dy + dz * dz) / (ax_ref[i] * ax_ref[i] + ay_ref[i] * ay_ref[i] + az_ref[i] * az_ref[i]));
        }

        std::sort(begin(delta), end(delta));

        EXPECT_TRUE(delta[numParticles * 0.99] < 3e-3);
        EXPECT_TRUE(delta[numParticles - 1] < 2e-2);
        if (utot_ref != 0.0)
            EXPECT_TRUE(std::abs(utot_ref - utot) / utot_ref < 1e-2);
        else
            EXPECT_NEAR(std::abs(utot_ref - utot), 0, 1e-4);

        V(1) printf("# %28s | %6i %5.2f %.15e %.15e %.15e %.15e %.15e\n", 
                test_name, numParticles, theta,
                delta[0],
                delta[numParticles / 2],
                delta[numParticles * 0.9],
                delta[numParticles * 0.99],
                delta[numParticles - 1]);
    }
}

TEST(Gravity, EwaldDisabled)
{
    using T             = double;
    using KeyType       = uint64_t;
    using MultipoleType = ryoanji::CartesianQuadrupole<T>;

    auto test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();

    float      G            = 1.0;
    LocalIndex numParticles = 100;
    std::vector<T> thetas   = {0.0, 0.5, 1.0};

    cstone::Box<T> box(-1, 1, cstone::BoundaryType::periodic);

    V(1) printf("# %28s | %6s %5s %21s\n", "Test", "nPart", "Theta", "min/50/10/1/max Error");
    V(1) printf("# %28s | %6s %5s %21s\n", "----", "-----", "-----", "---------------------");

    for (auto theta : thetas)
    {
        auto [coordinates, layout, octree, multipoles, centers, masses, h] = makeTestTree<T, KeyType, MultipoleType>(box, numParticles, theta);

        const T* x = coordinates.x().data();
        const T* y = coordinates.y().data();
        const T* z = coordinates.z().data();

        double         utot_ref;
        std::vector<T> ax_ref(numParticles, 0);
        std::vector<T> ay_ref(numParticles, 0);
        std::vector<T> az_ref(numParticles, 0);
        std::vector<T>  u_ref(numParticles, 0);

        {
        double hCut             = 0.0;
        double ewaldCut         = 0.0;
        double alpha_scale      = 0.0;
        int    numReplicaShells = 0;

        utot_ref = computeGravityEwald(octree.childOffsets.data(), octree.internalToLeaf.data(), octree.numLeafNodes,
                                     centers.data(), multipoles.data(), layout.data(), 0, octree.numLeafNodes, x, y, z,
                                     h.data(), masses.data(), G, ax_ref.data(), ay_ref.data(), az_ref.data(), 
                                     box, numReplicaShells, hCut, ewaldCut, alpha_scale);
        }

        double         utot;
        std::vector<T> ax(numParticles, 0);
        std::vector<T> ay(numParticles, 0);
        std::vector<T> az(numParticles, 0);
        {
        double hCut             = 2.8;
        double ewaldCut         = 2.6;
        double alpha_scale      = 2.0;
        int    numReplicaShells = 8;

        utot = computeGravityEwald(octree.childOffsets.data(), octree.internalToLeaf.data(), octree.numLeafNodes,
                                 centers.data(), multipoles.data(), layout.data(), 0, octree.numLeafNodes, x, y, z,
                                 h.data(), masses.data(), G, ax.data(), ay.data(), az.data(), 
                                 box, numReplicaShells, hCut, ewaldCut, alpha_scale, GRAV_NO_EWALD | GRAV_NO_REPLICAS);
        }
    
        // relative errors
        std::vector<T> delta(numParticles);
        for (LocalIndex i = 0; i < numParticles; ++i)
        {
            T dx = ax[i] - ax_ref[i];
            T dy = ay[i] - ay_ref[i];
            T dz = az[i] - az_ref[i];

            delta[i] = std::sqrt((dx * dx + dy * dy + dz * dz) / (ax_ref[i] * ax_ref[i] + ay_ref[i] * ay_ref[i] + az_ref[i] * az_ref[i]));
        }

        std::sort(begin(delta), end(delta));

        EXPECT_TRUE(delta[numParticles * 0.99] < 3e-10);
        EXPECT_TRUE(delta[numParticles - 1] < 2e-10);
        if (utot_ref != 0.0)
            EXPECT_TRUE(std::abs(utot_ref - utot) / utot_ref < 1e-10);
        else
            EXPECT_NEAR(std::abs(utot_ref - utot), 0, 1e-10);

        V(1) printf("  %28s | %6i %5.2f %.15e %.15e %.15e %.15e\n", 
                test_name, numParticles, theta,
                delta[0],
                delta[numParticles / 2],
                delta[numParticles * 0.9],
                delta[numParticles * 0.99],
                delta[numParticles - 1]);
    }
}

TEST(Gravity, EwaldOnlyReplicas)
{
    using T             = double;
    using KeyType       = uint64_t;
    using MultipoleType = ryoanji::CartesianQuadrupole<T>;

    auto test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();

    float      G            = 1.0;

    cstone::Box<T> box(-1, 1, cstone::BoundaryType::periodic);

    V(1) printf("# %28s | %6s %5s %9s %12s\n", "Test", "nPart", "Theta", "nReplicas", "Sum ax ay az");
    V(1) printf("# %28s | %6s %5s %9s %12s\n", "----", "-----", "-----", "---------", "------------");

    {
    LocalIndex numParticles = 100;
    auto theta              = 0.0;
    auto [coordinates, layout, octree, multipoles, centers, masses, h] = makeTestTree<T, KeyType, MultipoleType>(box, numParticles, theta, false);

    const T* x = coordinates.x().data();
    const T* y = coordinates.y().data();
    const T* z = coordinates.z().data();

    for (auto numReplicaShells = 0; numReplicaShells <= 8; numReplicaShells++)
    {
        double         utot;
        std::vector<T> ax(numParticles, 0);
        std::vector<T> ay(numParticles, 0);
        std::vector<T> az(numParticles, 0);
        {
        double hCut             = 2.8;
        double ewaldCut         = 2.6;
        double alpha_scale      = 2.0;

        utot = computeGravityEwald(octree.childOffsets.data(), octree.internalToLeaf.data(), octree.numLeafNodes,
                                 centers.data(), multipoles.data(), layout.data(), 0, octree.numLeafNodes, x, y, z,
                                 h.data(), masses.data(), G, ax.data(), ay.data(), az.data(), 
                                 box, numReplicaShells, hCut, ewaldCut, alpha_scale, GRAV_NO_EWALD);
        }

        T ax_tot = 0.0;
        T ay_tot = 0.0;
        T az_tot = 0.0;
        for (LocalIndex i = 0; i < numParticles; ++i)
        {
            ax_tot += ax[i];
            ay_tot += ay[i];
            az_tot += az[i];
        }

        EXPECT_NEAR(std::abs(ax_tot), 0, 1e-12);
        EXPECT_NEAR(std::abs(ay_tot), 0, 1e-12);
        EXPECT_NEAR(std::abs(az_tot), 0, 1e-12);

        V(1) printf("  %28s | %6i %5.2f %9i %23.15e %23.15e %23.15e\n", 
                test_name, numParticles, theta,
                numReplicaShells,
                ax_tot, ay_tot, az_tot);
    }
    }

    {
    LocalIndex numParticles = 1;
    auto theta              = 1.0;
    auto [coordinates, layout, octree, multipoles, centers, masses, h] = makeTestTree<T, KeyType, MultipoleType>(box, numParticles, theta, false);

    const T* x = coordinates.x().data();
    const T* y = coordinates.y().data();
    const T* z = coordinates.z().data();

    for (auto numReplicaShells = 0; numReplicaShells <= 8; numReplicaShells++)
    {
        double         utot;
        std::vector<T> ax(numParticles, 0);
        std::vector<T> ay(numParticles, 0);
        std::vector<T> az(numParticles, 0);
        {
        double hCut             = 2.8;
        double ewaldCut         = 2.6;
        double alpha_scale      = 2.0;

        utot = computeGravityEwald(octree.childOffsets.data(), octree.internalToLeaf.data(), octree.numLeafNodes,
                                 centers.data(), multipoles.data(), layout.data(), 0, octree.numLeafNodes, x, y, z,
                                 h.data(), masses.data(), G, ax.data(), ay.data(), az.data(), 
                                 box, numReplicaShells, hCut, ewaldCut, alpha_scale, GRAV_NO_EWALD);
        }

        T ax_tot = 0.0;
        T ay_tot = 0.0;
        T az_tot = 0.0;
        for (LocalIndex i = 0; i < numParticles; ++i)
        {
            ax_tot += ax[i];
            ay_tot += ay[i];
            az_tot += az[i];
        }

        EXPECT_NEAR(std::abs(ax_tot), 0, 1e-12);
        EXPECT_NEAR(std::abs(ay_tot), 0, 1e-12);
        EXPECT_NEAR(std::abs(az_tot), 0, 1e-12);

        V(1) printf("  %28s | %6i %5.2f %9i %23.15e %23.15e %23.15e\n", 
                test_name, numParticles, theta,
                numReplicaShells,
                ax_tot, ay_tot, az_tot);
    }
    }
}

TEST(Gravity, EwaldConvergedPotential)
{
    using T             = double;
    using KeyType       = uint64_t;
    using MultipoleType = ryoanji::CartesianQuadrupole<T>;

    auto test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();

    float      G            = 1.0;

    cstone::Box<T> box(-1, 1, cstone::BoundaryType::periodic);

    V(1) printf("# %28s | %6s %5s %6s %12s\n", "Test", "nPart", "Theta", "Mass", "Potential");
    V(1) printf("# %28s | %6s %5s %6s %12s\n", "----", "-----", "-----", "----", "---------");

    for (auto random_mass = 0; random_mass <= 1; random_mass++)
    for (LocalIndex numParticles = 1; numParticles <= 200000; numParticles *= 2)
    {
        auto theta            = 1.0;
        auto numReplicaShells = 1;

        auto [coordinates, layout, octree, multipoles, centers, masses, h] = makeTestTree<T, KeyType, MultipoleType>(box, numParticles, theta, random_mass == 1);

        const T* x = coordinates.x().data();
        const T* y = coordinates.y().data();
        const T* z = coordinates.z().data();

        {
            double         utot;
            std::vector<T> ax(numParticles, 0);
            std::vector<T> ay(numParticles, 0);
            std::vector<T> az(numParticles, 0);
            {
            double hCut             = 2.8;
            double ewaldCut         = 2.6;
            double alpha_scale      = 2.0;

            utot = computeGravityEwald(octree.childOffsets.data(), octree.internalToLeaf.data(), octree.numLeafNodes,
                                     centers.data(), multipoles.data(), layout.data(), 0, octree.numLeafNodes, x, y, z,
                                     h.data(), masses.data(), G, ax.data(), ay.data(), az.data(), 
                                     box, numReplicaShells, hCut, ewaldCut, alpha_scale);
            }

            V(1) printf("  %28s | %6i %5.2f %6s %23.15e\n", 
                    test_name, numParticles, theta,
                    random_mass==1 ? "random" : "const",
                    utot);
        }
    }

}

void runEwaldAlphaScaleTest(const float theta, const LocalIndex numParticles, 
        util::tuple<int,int,int> replica_range,
        util::tuple<int,int,int> ewald_shells,
        util::tuple<double,double,int> alpha_range,
        util::tuple<double,double> minCuts,
        const bool table_header = true)
{
    using T = double;
    using KeyType       = uint64_t;
    using MultipoleType = ryoanji::CartesianQuadrupole<T>;

    auto test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();

    auto [replicaStart, replicaEnd, replicaStep] = replica_range;
    auto [  alphaStart,   alphaEnd,   alphaStep] = alpha_range;
    auto [  ewaldStart,   ewaldEnd,   ewaldStep] = ewald_shells;

    auto [minhCut, minewaldCut] = minCuts;

    //auto minewaldCut = minCuts(1);
    auto steps        = (alphaStep == 0) ? 0 : std::max(alphaStep, 2);
    auto dalpha_scale = (alphaEnd - alphaStart) / (steps-1);

    const float G = 1.0;
    cstone::Box<T> box(-1, 1, cstone::BoundaryType::periodic);

    auto [coordinates, layout, octree, multipoles, centers, masses, h] = makeTestTree<T, KeyType, MultipoleType>(box, numParticles, theta);

    const T* x = coordinates.x().data();
    const T* y = coordinates.y().data();
    const T* z = coordinates.z().data();

    std::vector<T> ax(numParticles, 0);
    std::vector<T> ay(numParticles, 0);
    std::vector<T> az(numParticles, 0);

    if (table_header)
    {
        V(1) printf("# %6s %5s %6s %4s %5s %5s %5s %23s %23s %s\n", "nPart", "Theta", "Alpha", "Reps", "Ewald", "hCut", "eCut", "utot", "amag", "Sum ax ay az");
        V(1) printf("# %6s %5s %6s %4s %5s %5s %5s %23s %23s %s\n", "-----", "-----", "-----", "----", "-----", "----", "----", "----", "----", "------------");
    }

    for (auto i = 0; i < steps; i++)
    for (auto numShells      = replicaStart; numShells      <= replicaEnd; numShells      += replicaStep)
    for (auto numEwaldShells = ewaldStart;   numEwaldShells <= ewaldEnd;   numEwaldShells += ewaldStep)
    {
        std::fill(ax.begin(), ax.end(), 0);
        std::fill(ay.begin(), ay.end(), 0);
        std::fill(az.begin(), az.end(), 0);

        double alpha_scale = alphaStart + i * dalpha_scale;
        double hCut        = minhCut     + numShells + numEwaldShells;
        double ewaldCut    = minewaldCut + numShells + numEwaldShells;

        double utot = computeGravityEwald(octree.childOffsets.data(), octree.internalToLeaf.data(), octree.numLeafNodes,
                                     centers.data(), multipoles.data(), layout.data(), 0, octree.numLeafNodes, x, y, z,
                                     h.data(), masses.data(), G, ax.data(), ay.data(), az.data(), 
                                     box, numShells, hCut, ewaldCut, alpha_scale, GRAV_ALL);

        T axtot = std::accumulate(ax.begin(), ax.end(), 0.0);
        T aytot = std::accumulate(ay.begin(), ay.end(), 0.0);
        T aztot = std::accumulate(az.begin(), az.end(), 0.0);
        T amag  = std::sqrt(axtot*axtot + aytot*aytot + aztot*aztot);

        V(1) printf("  %6i %5.2f %6.2f %4i %5i %5.2f %5.2f %23.15e %23.15e %23.15e %23.15e %23.15e\n",
             numParticles, theta, alpha_scale, numShells, numEwaldShells, hCut, ewaldCut, utot, amag, axtot, aytot, aztot);
    }

}

TEST(Gravity, EwaldPartials)
{
    //GTEST_SKIP() << "Skipping EwaldPartials";

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

    runEwaldAlphaScaleTest(1.0,   1, {1,1,1}, {0,8,1}, {0.0,  2.0,  3}, {0.8, 0.6});
    runEwaldAlphaScaleTest(0.0, 100, {1,1,1}, {0,3,1}, {0.0,  2.0,  3}, {0.8, 0.6});
    runEwaldAlphaScaleTest(1.0, 100, {1,1,1}, {0,8,1}, {0.0,  2.0,  3}, {0.8, 0.6});
    runEwaldAlphaScaleTest(1.0, 100, {1,1,1}, {0,8,1}, {0.0, 16.0, 17}, {0.8, 0.6});
    runEwaldAlphaScaleTest(0.0, 100, {1,1,1}, {0,8,1}, {0.0, 16.0, 17}, {0.8, 0.6});

    runEwaldAlphaScaleTest(0.0, 100, {1,1,1}, {1,1,1}, {2.0, 2.0, 1}, {0.8, 0.6}, true);
    runEwaldAlphaScaleTest(0.2, 100, {1,1,1}, {1,1,1}, {2.0, 2.0, 1}, {0.8, 0.6}, false);
    runEwaldAlphaScaleTest(0.4, 100, {1,1,1}, {1,1,1}, {2.0, 2.0, 1}, {0.8, 0.6}, false);
    runEwaldAlphaScaleTest(0.5, 100, {1,1,1}, {1,1,1}, {2.0, 2.0, 1}, {0.8, 0.6}, false);
    runEwaldAlphaScaleTest(0.6, 100, {1,1,1}, {1,1,1}, {2.0, 2.0, 1}, {0.8, 0.6}, false);
    runEwaldAlphaScaleTest(0.8, 100, {1,1,1}, {1,1,1}, {2.0, 2.0, 1}, {0.8, 0.6}, false);
    runEwaldAlphaScaleTest(1.0, 100, {1,1,1}, {1,1,1}, {2.0, 2.0, 1}, {0.8, 0.6}, false);

//  runEwaldAlphaScaleTest(1.0, 2,   8, 0.0, 2.0, 3, cstone::Box<T>(-1, 1, cstone::BoundaryType::periodic));
//  //runEwaldAlphaScaleTest(0.0, 3,   8, 0.0, 2.0, 3, cstone::Box<T>(-1, 1, cstone::BoundaryType::periodic));
//  runEwaldAlphaScaleTest(0.0, 100,   3, 0.0, 2.0, 3, cstone::Box<T>(-1, 1, cstone::BoundaryType::periodic));
//  runEwaldAlphaScaleTest(1.0, 100,   8, 0.0, 2.0, 3, cstone::Box<T>(-1, 1, cstone::BoundaryType::periodic));

//  runEwaldAlphaScaleTest(1.0, 100,   8, 0.0, 16.0, 17, cstone::Box<T>(-1, 1, cstone::BoundaryType::periodic));
//  runEwaldAlphaScaleTest(0.0, 100,   8, 0.0, 16.0, 17, cstone::Box<T>(-1, 1, cstone::BoundaryType::periodic));

    //runEwaldAlphaScaleTest(1.0, 10000,   8, 0.0, 2.0, 3, cstone::Box<T>(-1, 1, cstone::BoundaryType::periodic));
    //runEwaldAlphaScaleTest(0.6, 10000,   8, 0.0, 2.0, 3, cstone::Box<T>(-1, 1, cstone::BoundaryType::periodic));
}








#if 0

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
    GTEST_SKIP() << "Skipping EwaldPartials";

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
#endif
