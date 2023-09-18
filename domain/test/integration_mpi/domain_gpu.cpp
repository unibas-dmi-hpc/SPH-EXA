/*
 * MIT License
 *
 * Copyright (c) 2022 CSCS, ETH Zurich
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
 * @brief Domain tests with n-ranks
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * Each rank creates identical random gaussian distributed particles.
 * Then each ranks grabs 1/n-th of those particles and uses them
 * to build the global domain, rejoining the same set of particles, but
 * distributed. Neighbors are then calculated for each local particle on each rank
 * and the total number of neighbors is summed up across all ranks.
 *
 * This neighbor sum is then compared against the neighbor sum obtained from the original
 * array that has all the global particles and tests that they match.
 *
 * This tests that the domain halo exchange finds all halos needed for a correct neighbor count.
 */

#include "gtest/gtest.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#define USE_CUDA

#include "coord_samples/random.hpp"
#include "cstone/domain/domain.hpp"

#include "cstone/util/reallocate.hpp"

using namespace cstone;

/*! @brief random gaussian coordinate init
 *
 * We're not using the coordinates from coord_samples, because we don't
 * want them sorted in Morton order.
 */
template<class T>
void initCoordinates(std::vector<T>& x, std::vector<T>& y, std::vector<T>& z, Box<T>& box, int rank)
{
    // std::random_device rd;
    std::mt19937 gen(rank);
    // random gaussian distribution at the center
    std::normal_distribution<T> disX((box.xmax() + box.xmin()) / 2, (box.xmax() - box.xmin()) / 5);
    std::normal_distribution<T> disY((box.ymax() + box.ymin()) / 2, (box.ymax() - box.ymin()) / 5);
    std::normal_distribution<T> disZ((box.zmax() + box.zmin()) / 2, (box.zmax() - box.zmin()) / 5);

    auto randX = [cmin = box.xmin(), cmax = box.xmax(), &disX, &gen]()
    { return std::max(std::min(disX(gen), cmax), cmin); };
    auto randY = [cmin = box.ymin(), cmax = box.ymax(), &disY, &gen]()
    { return std::max(std::min(disY(gen), cmax), cmin); };
    auto randZ = [cmin = box.zmin(), cmax = box.zmax(), &disZ, &gen]()
    { return std::max(std::min(disZ(gen), cmax), cmin); };

    std::generate(begin(x), end(x), randX);
    std::generate(begin(y), end(y), randY);
    std::generate(begin(z), end(z), randZ);
}

template<class KeyType, class T>
void randomGaussianAssignment(int rank, int numRanks)
{
    LocalIndex numParticles = 1000;
    Box<T> box(0, 1);
    int bucketSize      = 20;
    int bucketSizeFocus = 10;

    // Note: NOT sorted in SFC order
    std::vector<KeyType> keys(numParticles);
    std::vector<T> x(numParticles);
    std::vector<T> y(numParticles);
    std::vector<T> z(numParticles);
    std::vector<T> h(numParticles, 0.0000001);
    std::vector<T> m(numParticles, 1.0 / (numParticles * numRanks));
    initCoordinates(x, y, z, box, rank);

    thrust::device_vector<KeyType> d_keys;
    reallocateDevice(d_keys, numParticles, 1.0);
    thrust::device_vector<T> d_x = x;
    thrust::device_vector<T> d_y = y;
    thrust::device_vector<T> d_z = z;
    thrust::device_vector<T> d_h = h;
    thrust::device_vector<T> d_m = m;

    Domain<KeyType, T, CpuTag> domainCpu(rank, numRanks, bucketSize, bucketSizeFocus, 1.0, box);
    std::vector<T> hs1, hs2, hs3;
    domainCpu.sync(keys, x, y, z, h, std::tie(m), std::tie(hs1, hs2, hs3));

    Domain<KeyType, T, GpuTag> domainGpu(rank, numRanks, bucketSize, bucketSizeFocus, 1.0, box);
    thrust::device_vector<T> s1, s2, s3;
    domainGpu.sync(d_keys, d_x, d_y, d_z, d_h, std::tie(d_m), std::tie(s1, s2, s3));

    std::cout << "numHalos " << domainGpu.nParticlesWithHalos() - domainGpu.nParticles() << " cpu "
              << domainCpu.nParticlesWithHalos() - domainCpu.nParticles() << std::endl;

    ASSERT_EQ(domainCpu.nParticles(), domainGpu.nParticles());
    EXPECT_EQ(domainCpu.nParticlesWithHalos(), domainGpu.nParticlesWithHalos());
    EXPECT_EQ(domainCpu.globalTree().treeLeaves().size(), domainGpu.globalTree().treeLeaves().size());
    EXPECT_EQ(d_x.size(), x.size());

    {
        std::vector<KeyType> dl(d_keys.size());
        thrust::copy_n(d_keys.data(), d_keys.size(), dl.data());
        EXPECT_TRUE(std::equal(dl.begin(), dl.end(), keys.begin()));
    }
    {
        std::vector<T> dl(d_x.size());
        thrust::copy_n(d_x.data(), d_x.size(), dl.data());
        EXPECT_TRUE(std::equal(dl.begin(), dl.end(), x.begin()));
    }

    {
        auto centers = domainCpu.focusTree().expansionCenters();
        for (auto ci : centers[0])
            std::cout << ci << " ";
        std::cout << std::endl;
    }
    {
        auto centers = domainGpu.focusTree().expansionCenters();
        for (auto ci : centers[0])
            std::cout << ci << " ";
        std::cout << std::endl;
    }
}

TEST(DomainGpu, matchTreeCpu)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    randomGaussianAssignment<uint64_t, double>(rank, numRanks);
}

TEST(DomainGpu, reapplySync)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    using Real    = double;
    using KeyType = unsigned;

    Box<Real> box(0, 1);
    LocalIndex numParticlesPerRank = 10000;
    unsigned bucketSize            = 1024;
    unsigned bucketSizeFocus       = 8;
    float theta                    = 0.5;

    // Note: rank used as seed, so each rank will get different coordinates
    RandomCoordinates<Real, SfcKind<KeyType>> coordinates(numParticlesPerRank, box, rank);

    std::vector<Real> x(coordinates.x().begin(), coordinates.x().end());
    std::vector<Real> y(coordinates.y().begin(), coordinates.y().end());
    std::vector<Real> z(coordinates.z().begin(), coordinates.z().end());
    std::vector<Real> h(numParticlesPerRank, 0.1 / std::cbrt(numRanks));
    std::vector<KeyType> keys(x.size());

    thrust::device_vector<Real> d_x       = x;
    thrust::device_vector<Real> d_y       = y;
    thrust::device_vector<Real> d_z       = z;
    thrust::device_vector<Real> d_h       = h;
    thrust::device_vector<KeyType> d_keys = keys;

    Domain<KeyType, Real, GpuTag> domain(rank, numRanks, bucketSize, bucketSizeFocus, theta, box);

    thrust::device_vector<Real> s1, s2, gpuOrdering;
    domain.sync(d_keys, d_x, d_y, d_z, d_h, std::tuple{}, std::tie(s1, s2, gpuOrdering));

    // modify coordinates
    {
        RandomCoordinates<Real, SfcKind<KeyType>> scord(domain.nParticles(), box, numRanks + rank);
        thrust::copy(scord.x().begin(), scord.x().end(), d_x.begin() + domain.startIndex());
        thrust::copy(scord.y().begin(), scord.y().end(), d_y.begin() + domain.startIndex());
        thrust::copy(scord.z().begin(), scord.z().end(), d_z.begin() + domain.startIndex());
    }

    std::vector<Real> host_property(d_x.size());
    for (size_t i = 0; i < x.size(); ++i)
    {
        host_property[i] = numParticlesPerRank * rank + i;
    }
    thrust::device_vector<Real> property = host_property;

    // exchange property together with sync
    domain.sync(d_keys, d_x, d_y, d_z, d_h, std::tie(property), std::tie(s1, s2, gpuOrdering));

    std::vector<Real> hs1, hs2;
    domain.reapplySync(std::tie(host_property), hs1, hs2, gpuOrdering);

    EXPECT_EQ(property.size(), host_property.size());

    thrust::host_vector<Real> dl_property = property;

    int numPass = 0;
    for (int i = domain.startIndex(); i < domain.endIndex(); ++i)
    {
        if (dl_property[i] == host_property[i]) numPass++;
    }
    EXPECT_EQ(numPass, domain.nParticles());

    {
        std::vector<Real> a(dl_property.begin() + domain.startIndex(), dl_property.begin() + domain.endIndex());
        std::vector<Real> b(host_property.begin() + domain.startIndex(), host_property.begin() + domain.endIndex());
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        std::vector<Real> s(a.size());
        auto it       = std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), s.begin());
        int numCommon = it - s.begin();
        EXPECT_EQ(numCommon, domain.nParticles());
    }
}
