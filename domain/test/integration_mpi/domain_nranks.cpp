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

/*! \file
 * \brief Domain tests with n-ranks
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * Each rank creates identical random gaussian distributed particles.
 * Then each ranks grabs 1/n-th of those particles and uses them
 * to build the global domain, rejoining the same set of particles, but
 * distributed. Neighbors are then calculated for each local particle on each rank
 * and the total number of neighbors is summed up across all all ranks.
 *
 * This neighbor sum is then compared again the neighbor sum obtained from the original
 * array that has all the global particles and tests that they match.
 *
 * This tests that the domain halo exchange finds all halos needed for a correct neighbor count.
 */

#include "gtest/gtest.h"

#include "coord_samples/random.hpp"
#include "cstone/domain.hpp"
#include "cstone/findneighbors.hpp"

using namespace cstone;

//! \brief simple N^2 all-to-all neighbor search
template<class T>
static void findNeighborsNaive(int i, const T* x, const T* y, const T* z, const T* h, int n,
                               int *neighbors, int *neighborsCount, int ngmax)
{
    T r2 = h[i] * h[i];

    T xi = x[i], yi = y[i], zi = z[i];

    int ngcount = 0;
    for (int j = 0; j < n; ++j)
    {
        if (j == i) { continue; }
        // i only interacts with j if j also interacts with i
        T r2mutual = std::min(h[j] * h[j], r2);
        if (ngcount < ngmax && distancesq(xi, yi, zi, x[j], y[j], z[j]) < r2mutual)
        {
            neighbors[ngcount++] = j;
        }
    }
    *neighborsCount = ngcount;
}


template<class T>
void initCoordinates(std::vector<T>& x, std::vector<T>& y, std::vector<T>& z, Box<T>& box)
{
    //std::random_device rd;
    std::mt19937 gen(42);
    // random gaussian distribution at the center
    std::normal_distribution<T> disX((box.xmax() + box.xmin())/2, (box.xmax() - box.xmin())/5);
    std::normal_distribution<T> disY((box.ymax() + box.ymin())/2, (box.ymax() - box.ymin())/5);
    std::normal_distribution<T> disZ((box.zmax() + box.zmin())/2, (box.zmax() - box.zmin())/5);

    auto randX = [cmin=box.xmin(), cmax=box.xmax(), &disX, &gen]() { return std::max(std::min(disX(gen), cmax), cmin); };
    auto randY = [cmin=box.ymin(), cmax=box.ymax(), &disY, &gen]() { return std::max(std::min(disY(gen), cmax), cmin); };
    auto randZ = [cmin=box.zmin(), cmax=box.zmax(), &disZ, &gen]() { return std::max(std::min(disZ(gen), cmax), cmin); };

    std::generate(begin(x), end(x), randX);
    std::generate(begin(y), end(y), randY);
    std::generate(begin(z), end(z), randZ);
}

template<class I, class T>
void randomGaussianDomain(int rank, int nRanks)
{
    int    nParticles    = 1000;
    T      smoothingLength = 0.02;
    Box<T> box{-1, 1};
    int    bucketSize = 10;
    nParticles = (nParticles/nRanks) * nRanks;

    // nParticles identical coordinates on each rank
    std::vector<T> xGlobal(nParticles);
    std::vector<T> yGlobal(nParticles);
    std::vector<T> zGlobal(nParticles);
    initCoordinates(xGlobal, yGlobal, zGlobal, box);

    std::vector<T> hGlobal(nParticles, smoothingLength);

    for (int i = 0; i < hGlobal.size(); ++i)
    {
        hGlobal[i] = smoothingLength * (0.2 + 30*(xGlobal[i] * xGlobal[i] + yGlobal[i] * yGlobal[i] + zGlobal[i] * zGlobal[i]));
    }

    int nParticlesPerRank = nParticles / nRanks;

    std::vector<T> x{xGlobal.begin() + rank*nParticlesPerRank, xGlobal.begin() + (rank+1)*nParticlesPerRank};
    std::vector<T> y{yGlobal.begin() + rank*nParticlesPerRank, yGlobal.begin() + (rank+1)*nParticlesPerRank};
    std::vector<T> z{zGlobal.begin() + rank*nParticlesPerRank, zGlobal.begin() + (rank+1)*nParticlesPerRank};
    std::vector<T> h{hGlobal.begin() + rank*nParticlesPerRank, hGlobal.begin() + (rank+1)*nParticlesPerRank};

    Domain<I, T> domain(rank, nRanks, bucketSize);

    std::vector<I> codes;
    domain.sync(x, y, z, h, codes);

    int localCount = domain.endIndex() - domain.startIndex();
    int localCountSum = localCount;
    int extractedCount = x.size();
    MPI_Allreduce(MPI_IN_PLACE, &localCountSum, 1, MpiType<int>{}, MPI_SUM, MPI_COMM_WORLD);
    EXPECT_EQ(localCountSum, nParticles);

    // the actual box is not exactly [-1,1], but something very slightly smaller
    T xmin = *std::min_element(begin(xGlobal), end(xGlobal));
    T xmax = *std::max_element(begin(xGlobal), end(xGlobal));
    T ymin = *std::min_element(begin(yGlobal), end(yGlobal));
    T ymax = *std::max_element(begin(yGlobal), end(yGlobal));
    T zmin = *std::min_element(begin(zGlobal), end(zGlobal));
    T zmax = *std::max_element(begin(zGlobal), end(zGlobal));
    Box<T> actualBox{xmin, xmax, ymin, ymax, zmin, zmax};
    std::vector<I> mortonCodes(x.size());
    computeMortonCodes(begin(x), end(x), begin(y), begin(z), begin(mortonCodes), actualBox);

    // check that particles are Morton order sorted and the codes are in sync with the x,y,z arrays
    EXPECT_EQ(mortonCodes, codes);
    for (int i = 0; i < mortonCodes.size()-1; ++i)
    {
        EXPECT_TRUE(mortonCodes[i] <= mortonCodes[i+1]);
    }

    int ngmax = 200;
    std::vector<int> neighbors(localCount * ngmax);
    std::vector<int> neighborsCount(localCount);
    for (int i = 0; i < localCount; ++i)
    {
        int particleIndex = i + domain.startIndex();
        //findNeighbors(particleIndex, x.data(), y.data(), z.data(), h[particleIndex], actualBox.xmax()-actualBox.xmin(),
        //              mortonCodes.data(), neighbors.data(), neighborsCount.data(), localCount, ngmax);
        findNeighborsNaive(particleIndex, x.data(), y.data(), z.data(), h.data(), extractedCount,
                           neighbors.data() + i*ngmax, neighborsCount.data() + i, ngmax);
    }

    int neighborSum = std::accumulate(begin(neighborsCount), end(neighborsCount), 0);
    MPI_Allreduce(MPI_IN_PLACE, &neighborSum, 1, MpiType<int>{}, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0)
    {
        std::cout << " neighborSum " << neighborSum << std::endl;
        //std::cout << "localCount " << localCount << " " << std::endl;
        //std::cout << "extractedCount " << extractedCount << std::endl;
    }

    {
        // calculate reference neighbor sum from the full arrays
        std::vector<int> neighborsRef(nParticles*ngmax);
        std::vector<int> neighborsCountRef(nParticles);
        for (int i = 0; i < nParticles; ++i)
        {
            //findNeighbors(i, xGlobal.data(), yGlobal.data(), zGlobal.data(), h[i], actualBox.xmax()-actualBox.xmin(),
            //              mortonCodes.data(), neighborsRef.data(), neighborsCountRef.data(), nParticles, ngmax);
            findNeighborsNaive(i, xGlobal.data(), yGlobal.data(), zGlobal.data(), hGlobal.data(), nParticles,
                               neighborsRef.data() + i*ngmax, neighborsCountRef.data() + i, ngmax);
        }
        int neighborSumRef = std::accumulate(begin(neighborsCountRef), end(neighborsCountRef), 0);
        EXPECT_EQ(neighborSum, neighborSumRef);
    }
}

TEST(Domain, randomGaussianNeighborSum)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    randomGaussianDomain<unsigned, double>(rank, nRanks);
    randomGaussianDomain<uint64_t, double>(rank, nRanks);
    randomGaussianDomain<unsigned, float>(rank, nRanks);
    randomGaussianDomain<uint64_t, float>(rank, nRanks);
}
