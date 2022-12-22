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
 * @brief Integration test of Ewald periodic boundary conditions 
 *
 * @author Jonathan Coles        <jonathan.coles@cscs.ch>
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
    [[maybe_unused]] auto test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    //GTEST_SKIP() << "Skipping " << test_name;

    EXPECT_EQ(   EWALD_GRAVITY_SWITCH::GRAV_NO_EWALD
               + EWALD_GRAVITY_SWITCH::GRAV_NO_REPLICAS
            ,    EWALD_GRAVITY_SWITCH::GRAV_NO_EWALD
               | EWALD_GRAVITY_SWITCH::GRAV_NO_REPLICAS);
}

TEST(Gravity, EwaldBaseline)
{
    [[maybe_unused]] auto test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    //GTEST_SKIP() << "Skipping " << test_name;

    using T             = double;
    using KeyType       = uint64_t;
    using MultipoleType = ryoanji::CartesianQuadrupole<T>;

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
    [[maybe_unused]] auto test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    //GTEST_SKIP() << "Skipping " << test_name;

    using T             = double;
    using KeyType       = uint64_t;
    using MultipoleType = ryoanji::CartesianQuadrupole<T>;

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

        V(1) printf("  %28s | %6i %5.2f %.15e %.15e %.15e %.15e %.15e\n", 
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
    [[maybe_unused]] auto test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    //GTEST_SKIP() << "Skipping " << test_name;

    using T             = double;
    using KeyType       = uint64_t;
    using MultipoleType = ryoanji::CartesianQuadrupole<T>;

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
    [[maybe_unused]] auto test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    //GTEST_SKIP() << "Skipping " << test_name;

    using T             = double;
    using KeyType       = uint64_t;
    using MultipoleType = ryoanji::CartesianQuadrupole<T>;

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
    [[maybe_unused]] auto test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    //GTEST_SKIP() << "Skipping " << test_name;

    using T = double;
    using KeyType       = uint64_t;
    using MultipoleType = ryoanji::CartesianQuadrupole<T>;

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
    [[maybe_unused]] auto test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    //GTEST_SKIP() << "Skipping " << test_name;

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
}
