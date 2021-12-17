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

#include "coord_samples/random.hpp"
#include "cstone/domain/domain.hpp"
#include "cstone/findneighbors.hpp"

using namespace cstone;


/*! @brief random gaussian coordinate init
 *
 * We're not using the coordinates from coord_samples, because we don't
 * want them sorted in Morton order.
 */
template<class T>
void initCoordinates(std::vector<T>& x, std::vector<T>& y, std::vector<T>& z, Box<T>& box)
{
    // std::random_device rd;
    std::mt19937 gen(42);
    // random gaussian distribution at the center
    std::normal_distribution<T> disX((box.xmax() + box.xmin()) / 2, (box.xmax() - box.xmin()) / 5);
    std::normal_distribution<T> disY((box.ymax() + box.ymin()) / 2, (box.ymax() - box.ymin()) / 5);
    std::normal_distribution<T> disZ((box.zmax() + box.zmin()) / 2, (box.zmax() - box.zmin()) / 5);

    auto randX = [cmin = box.xmin(), cmax = box.xmax(), &disX, &gen]() {
        return std::max(std::min(disX(gen), cmax), cmin);
    };
    auto randY = [cmin = box.ymin(), cmax = box.ymax(), &disY, &gen]() {
        return std::max(std::min(disY(gen), cmax), cmin);
    };
    auto randZ = [cmin = box.zmin(), cmax = box.zmax(), &disZ, &gen]() {
        return std::max(std::min(disZ(gen), cmax), cmin);
    };

    std::generate(begin(x), end(x), randX);
    std::generate(begin(y), end(y), randY);
    std::generate(begin(z), end(z), randZ);
}

template<class KeyType, class T, class DomainType>
void randomGaussianDomain(DomainType domain, int rank, int nRanks, bool equalizeH = false)
{
    LocalParticleIndex numParticles = (1000 / nRanks) * nRanks;
    Box<T> box = domain.box();

    // numParticles identical coordinates on each rank
    // Note: NOT sorted in morton order
    std::vector<T> xGlobal(numParticles);
    std::vector<T> yGlobal(numParticles);
    std::vector<T> zGlobal(numParticles);
    initCoordinates(xGlobal, yGlobal, zGlobal, box);

    std::vector<T> hGlobal(numParticles, 0.1);

    if (!equalizeH)
    {
        for (std::size_t i = 0; i < hGlobal.size(); ++i)
        {
            // tuned such that the particles far from the center have a bigger radius to compensate for lower density
            hGlobal[i] = 0.05 + 0.2 * (xGlobal[i] * xGlobal[i] + yGlobal[i] * yGlobal[i] + zGlobal[i] * zGlobal[i]);
        }
    }

    LocalParticleIndex nParticlesPerRank = numParticles / nRanks;

    std::vector<T> x{xGlobal.begin() + rank * nParticlesPerRank, xGlobal.begin() + (rank + 1) * nParticlesPerRank};
    std::vector<T> y{yGlobal.begin() + rank * nParticlesPerRank, yGlobal.begin() + (rank + 1) * nParticlesPerRank};
    std::vector<T> z{zGlobal.begin() + rank * nParticlesPerRank, zGlobal.begin() + (rank + 1) * nParticlesPerRank};
    std::vector<T> h{hGlobal.begin() + rank * nParticlesPerRank, hGlobal.begin() + (rank + 1) * nParticlesPerRank};

    std::vector<KeyType> codes(x.size());
    domain.sync(x, y, z, h, codes);

    LocalParticleIndex localCount = domain.endIndex() - domain.startIndex();
    LocalParticleIndex localCountSum = localCount;
    //int extractedCount = x.size();
    MPI_Allreduce(MPI_IN_PLACE, &localCountSum, 1, MpiType<int>{}, MPI_SUM, MPI_COMM_WORLD);
    EXPECT_EQ(localCountSum, numParticles);

    // box got updated if not using PBC
    box = domain.box();
    std::vector<KeyType> particleKeys(x.size());
    computeSfcKeys(x.data(), y.data(), z.data(), sfcKindPointer(particleKeys.data()), x.size(), box);

    // check that particles are Morton order sorted and the codes are in sync with the x,y,z arrays
    EXPECT_EQ(particleKeys, codes);
    EXPECT_TRUE(std::is_sorted(begin(particleKeys), end(particleKeys)));

    int ngmax = 300;
    std::vector<int> neighbors(localCount * ngmax);
    std::vector<int> neighborsCount(localCount);
    findNeighbors(x.data(), y.data(), z.data(), h.data(), domain.startIndex(), domain.endIndex(), x.size(),
                  box, sfcKindPointer(particleKeys.data()), neighbors.data(), neighborsCount.data(), ngmax);

    int neighborSum = std::accumulate(begin(neighborsCount), end(neighborsCount), 0);
    MPI_Allreduce(MPI_IN_PLACE, &neighborSum, 1, MpiType<int>{}, MPI_SUM, MPI_COMM_WORLD);

    {
        // Note: global coordinates are not yet in Morton order
        std::vector<KeyType> codesGlobal(numParticles);
        computeSfcKeys(xGlobal.data(), yGlobal.data(), zGlobal.data(), sfcKindPointer(codesGlobal.data()),
                       numParticles, box);
        std::vector<LocalParticleIndex> ordering(numParticles);
        std::iota(begin(ordering), end(ordering), LocalParticleIndex(0));
        sort_by_key(begin(codesGlobal), end(codesGlobal), begin(ordering));
        reorderInPlace(ordering, xGlobal.data());
        reorderInPlace(ordering, yGlobal.data());
        reorderInPlace(ordering, zGlobal.data());
        reorderInPlace(ordering, hGlobal.data());

        // calculate reference neighbor sum from the full arrays
        std::vector<int> neighborsRef(numParticles * ngmax);
        std::vector<int> neighborsCountRef(numParticles);
        findNeighbors(xGlobal.data(), yGlobal.data(), zGlobal.data(), hGlobal.data(), 0, numParticles,
                      numParticles, box, sfcKindPointer(codesGlobal.data()), neighborsRef.data(),
                      neighborsCountRef.data(), ngmax);

        int neighborSumRef = std::accumulate(begin(neighborsCountRef), end(neighborsCountRef), 0);
        EXPECT_EQ(neighborSum, neighborSumRef);
    }
}


TEST(FocusDomain, randomGaussianNeighborSum)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    int bucketSize = 50;
    int bucketSizeFocus = 10;
    float theta = 1.0;

    {
        Domain<unsigned, double> domain(rank, nRanks, bucketSize, bucketSizeFocus, theta, {-1, 1});
        randomGaussianDomain<unsigned, double>(domain, rank, nRanks);
    }
    {
        Domain<uint64_t, double> domain(rank, nRanks, bucketSize, bucketSizeFocus, theta, {-1, 1});
        randomGaussianDomain<uint64_t, double>(domain, rank, nRanks);
    }
    {
        Domain<unsigned, float> domain(rank, nRanks, bucketSize, bucketSizeFocus, theta, {-1, 1});
        randomGaussianDomain<unsigned, float>(domain, rank, nRanks);
    }
    {
        Domain<uint64_t, float> domain(rank, nRanks, bucketSize, bucketSizeFocus, theta, {-1, 1});
        randomGaussianDomain<uint64_t, float>(domain, rank, nRanks);
    }
}

TEST(FocusDomain, randomGaussianNeighborSumPbc)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    int bucketSize = 50;
    int bucketSizeFocus = 10;
    float theta = 1.0;

    {
        Domain<unsigned, double> domain(rank, nRanks, bucketSize, bucketSizeFocus, theta, {-1, 1, true});
        randomGaussianDomain<unsigned, double>(domain, rank, nRanks);
    }
    {
        Domain<uint64_t, double> domain(rank, nRanks, bucketSize, bucketSizeFocus, theta, {-1, 1, true});
        randomGaussianDomain<uint64_t, double>(domain, rank, nRanks);
    }
    {
        Domain<unsigned, float> domain(rank, nRanks, bucketSize, bucketSizeFocus, theta, {-1, 1, true});
        randomGaussianDomain<unsigned, float>(domain, rank, nRanks);
    }
    {
        Domain<uint64_t, float> domain(rank, nRanks, bucketSize, bucketSizeFocus, theta, {-1, 1, true});
        randomGaussianDomain<uint64_t, float>(domain, rank, nRanks);
    }
}

TEST(FocusDomain, assignmentShift)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    using Real = double;
    using KeyType = unsigned;

    Box<Real> box(0, 1);
    LocalParticleIndex numParticlesPerRank = 15000;
    unsigned bucketSize = 1024;
    unsigned bucketSizeFocus = 8;
    float theta = 1.0;

    RandomCoordinates<Real, SfcKind<KeyType>> coordinates(numParticlesPerRank, box, rank);

    std::vector<Real> x(coordinates.x().begin(), coordinates.x().end());
    std::vector<Real> y(coordinates.y().begin(), coordinates.y().end());
    std::vector<Real> z(coordinates.z().begin(), coordinates.z().end());
    std::vector<Real> h(numParticlesPerRank, 0.1);

    Domain<KeyType, Real> domain(rank, numRanks, bucketSize, bucketSizeFocus, theta, box);

    std::vector<KeyType> particleKeys(x.size());

    domain.sync(x,y,z,h, particleKeys);

    if (rank == 2)
    {
        for (int k = 0; k < 700; ++k)
        {
            x[k + domain.startIndex()] -= 0.25;
        }
    }

    domain.sync(x,y,z,h, particleKeys);

    std::vector<Real> property(domain.nParticlesWithHalos(), -1);
    for (LocalParticleIndex i = domain.startIndex(); i < domain.endIndex(); ++i)
    {
        property[i] = rank;
    }

    domain.exchangeHalos(property);

    EXPECT_TRUE(std::count(property.begin(), property.end(), -1) == 0);
    EXPECT_TRUE(std::count(property.begin(), property.end(), rank) == domain.nParticles());
}
