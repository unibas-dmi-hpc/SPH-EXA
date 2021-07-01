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
#include "cstone/domain/domain_focus.hpp"
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

template<class I, class T, class DomainType>
void randomGaussianDomain(DomainType domain, int rank, int nRanks, bool equalizeH = false)
{
    int nParticles = (1000 / nRanks) * nRanks;
    Box<T> box = domain.box();

    // nParticles identical coordinates on each rank
    // Note: NOT sorted in morton order
    std::vector<T> xGlobal(nParticles);
    std::vector<T> yGlobal(nParticles);
    std::vector<T> zGlobal(nParticles);
    initCoordinates(xGlobal, yGlobal, zGlobal, box);

    std::vector<T> hGlobal(nParticles, 0.1);

    if (!equalizeH)
    {
        for (std::size_t i = 0; i < hGlobal.size(); ++i)
        {
            // tuned such that the particles far from the center have a bigger radius to compensate for lower density
            hGlobal[i] = 0.05 + 0.2 * (xGlobal[i] * xGlobal[i] + yGlobal[i] * yGlobal[i] + zGlobal[i] * zGlobal[i]);
        }
    }

    int nParticlesPerRank = nParticles / nRanks;

    std::vector<T> x{xGlobal.begin() + rank * nParticlesPerRank, xGlobal.begin() + (rank + 1) * nParticlesPerRank};
    std::vector<T> y{yGlobal.begin() + rank * nParticlesPerRank, yGlobal.begin() + (rank + 1) * nParticlesPerRank};
    std::vector<T> z{zGlobal.begin() + rank * nParticlesPerRank, zGlobal.begin() + (rank + 1) * nParticlesPerRank};
    std::vector<T> h{hGlobal.begin() + rank * nParticlesPerRank, hGlobal.begin() + (rank + 1) * nParticlesPerRank};

    std::vector<I> codes;
    domain.sync(x, y, z, h, codes);

    int localCount = domain.endIndex() - domain.startIndex();
    int localCountSum = localCount;
    int extractedCount = x.size();
    MPI_Allreduce(MPI_IN_PLACE, &localCountSum, 1, MpiType<int>{}, MPI_SUM, MPI_COMM_WORLD);
    EXPECT_EQ(localCountSum, nParticles);

    // box got updated if not using PBC
    box = domain.box();
    std::vector<I> mortonCodes(x.size());
    computeMortonCodes(begin(x), end(x), begin(y), begin(z), begin(mortonCodes), box);

    // check that particles are Morton order sorted and the codes are in sync with the x,y,z arrays
    EXPECT_EQ(mortonCodes, codes);
    EXPECT_TRUE(std::is_sorted(begin(mortonCodes), end(mortonCodes)));

    int ngmax = 300;
    std::vector<int> neighbors(localCount * ngmax);
    std::vector<int> neighborsCount(localCount);
    for (int i = 0; i < localCount; ++i)
    {
        int particleIndex = i + domain.startIndex();
        findNeighbors(particleIndex, x.data(), y.data(), z.data(), h.data(), box,
                      mortonCodes.data(), neighbors.data() + i * ngmax, neighborsCount.data() + i,
                      extractedCount, ngmax);
    }

    int neighborSum = std::accumulate(begin(neighborsCount), end(neighborsCount), 0);
    MPI_Allreduce(MPI_IN_PLACE, &neighborSum, 1, MpiType<int>{}, MPI_SUM, MPI_COMM_WORLD);
    //if (rank == 0)
    //{
    //    std::cout << " neighborSum " << neighborSum << std::endl;
    //    std::cout << "localCount " << localCount << " " << std::endl;
    //    std::cout << "extractedCount " << extractedCount << std::endl;
    //}

    {
        // Note: global coordinates are not yet in Morton order
        std::vector<I> codesGlobal(nParticles);
        computeMortonCodes(begin(xGlobal), end(xGlobal), begin(yGlobal), begin(zGlobal), begin(codesGlobal), box);
        std::vector<LocalParticleIndex> ordering(nParticles);
        std::iota(begin(ordering), end(ordering), LocalParticleIndex(0));
        sort_by_key(begin(codesGlobal), end(codesGlobal), begin(ordering));
        reorder(ordering, xGlobal);
        reorder(ordering, yGlobal);
        reorder(ordering, zGlobal);
        reorder(ordering, hGlobal);

        // calculate reference neighbor sum from the full arrays
        std::vector<int> neighborsRef(nParticles * ngmax);
        std::vector<int> neighborsCountRef(nParticles);
        for (int i = 0; i < nParticles; ++i)
        {
            findNeighbors(i, xGlobal.data(), yGlobal.data(), zGlobal.data(), hGlobal.data(), box,
                          codesGlobal.data(), neighborsRef.data() + i * ngmax, neighborsCountRef.data() + i,
                          nParticles, ngmax);
        }

        int neighborSumRef = std::accumulate(begin(neighborsCountRef), end(neighborsCountRef), 0);
        EXPECT_EQ(neighborSum, neighborSumRef);
    }
}


/*! @brief global-tree based domain with PBC
 *
  * This test case and the one below are affected by the mutuality limitation of findHalos based on
  * the global tree, where two nodes i and j are only halos if i is a halo of j
  * AND vice versa. This leads to two missing halo particles for at least some values of numRanks between 2 and 5.
  * The focus-tree based domain overcomes this limitation.
 */
TEST(Domain, randomGaussianNeighborSum)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    int bucketSize = 10;
    // avoid halo mutuality limitation
    bool equalizeH = true;

    {
        Domain<unsigned, double> domain(rank, nRanks, bucketSize, {-1, 1});
        randomGaussianDomain<unsigned, double>(domain, rank, nRanks, equalizeH);
    }
    {
        Domain<uint64_t, double> domain(rank, nRanks, bucketSize, {-1, 1});
        randomGaussianDomain<uint64_t, double>(domain, rank, nRanks, equalizeH);
    }
    {
        Domain<unsigned, float> domain(rank, nRanks, bucketSize, {-1, 1});
        randomGaussianDomain<unsigned, float>(domain, rank, nRanks, equalizeH);
    }
    {
        Domain<uint64_t, float> domain(rank, nRanks, bucketSize, {-1, 1});
        randomGaussianDomain<uint64_t, float>(domain, rank, nRanks, equalizeH);
    }
}

TEST(Domain, randomGaussianNeighborSumPbc)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    int bucketSize = 10;
    // avoid halo mutuality limitation
    bool equalizeH = true;

    {
        Domain<unsigned, double> domain(rank, nRanks, bucketSize, {-1, 1, true});
        randomGaussianDomain<unsigned, double>(domain, rank, nRanks, equalizeH);
    }
    {
        Domain<uint64_t, double> domain(rank, nRanks, bucketSize, {-1, 1, true});
        randomGaussianDomain<uint64_t, double>(domain, rank, nRanks, equalizeH);
    }

    {
        Domain<unsigned, float> domain(rank, nRanks, bucketSize, {-1, 1, true});
        randomGaussianDomain<unsigned, float>(domain, rank, nRanks, equalizeH);
    }
    {
        Domain<uint64_t, float> domain(rank, nRanks, bucketSize, {-1, 1, true});
        randomGaussianDomain<uint64_t, float>(domain, rank, nRanks, equalizeH);
    }
}

TEST(FocusDomain, randomGaussianNeighborSum)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    int bucketSize = 50;
    int bucketSizeFocus = 10;

    {
        FocusedDomain<unsigned, double> domain(rank, nRanks, bucketSize, bucketSizeFocus, {-1, 1});
        randomGaussianDomain<unsigned, double>(domain, rank, nRanks);
    }
    {
        FocusedDomain<uint64_t, double> domain(rank, nRanks, bucketSize, bucketSizeFocus, {-1, 1});
        randomGaussianDomain<uint64_t, double>(domain, rank, nRanks);
    }
    {
        FocusedDomain<unsigned, float> domain(rank, nRanks, bucketSize, bucketSizeFocus, {-1, 1});
        randomGaussianDomain<unsigned, float>(domain, rank, nRanks);
    }
    {
        FocusedDomain<uint64_t, float> domain(rank, nRanks, bucketSize, bucketSizeFocus, {-1, 1});
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

    {
        FocusedDomain<unsigned, double> domain(rank, nRanks, bucketSize, bucketSizeFocus, {-1, 1, true});
        randomGaussianDomain<unsigned, double>(domain, rank, nRanks);
    }
    {
        FocusedDomain<uint64_t, double> domain(rank, nRanks, bucketSize, bucketSizeFocus, {-1, 1, true});
        randomGaussianDomain<uint64_t, double>(domain, rank, nRanks);
    }
    {
        FocusedDomain<unsigned, float> domain(rank, nRanks, bucketSize, bucketSizeFocus, {-1, 1, true});
        randomGaussianDomain<unsigned, float>(domain, rank, nRanks);
    }
    {
        FocusedDomain<uint64_t, float> domain(rank, nRanks, bucketSize, bucketSizeFocus, {-1, 1, true});
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

    RandomCoordinates<Real, KeyType> coordinates(numParticlesPerRank, box, rank);

    std::vector<Real> x = coordinates.x();
    std::vector<Real> y = coordinates.y();
    std::vector<Real> z = coordinates.z();
    std::vector<Real> h(numParticlesPerRank, 0.1);

    FocusedDomain<KeyType, Real> domain(rank, numRanks, bucketSize, bucketSizeFocus, box);

    std::vector<KeyType> particleKeys;

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