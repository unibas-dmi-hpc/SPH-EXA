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
 * \brief Various domain tests with 2 ranks
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "sfc/domain.hpp"

using namespace sphexa;

template<class I, class T>
void noHalos(int rank, int nRanks)
{
    int bucketSize = 1;
    Domain<I, T> domain(rank, nRanks, bucketSize);

    std::vector<T> x{0.5, 0.6};
    std::vector<T> y{0.5, 0.6};
    std::vector<T> z{0.5, 0.6};
    // radii around 0.5 and 0.6 don't overlap
    std::vector<T> h{0.05, 0.05};

    domain.sync(x,y,z,h);

    EXPECT_EQ(domain.startIndex(), 0);
    EXPECT_EQ(domain.endIndex(), 2);

    std::vector<T> cref;
    if (rank == 0)
        cref = std::vector<T>{0.5, 0.5};
    else if (rank == 1)
        cref = std::vector<T>{0.6, 0.6};

    EXPECT_EQ(cref, x);
    EXPECT_EQ(cref, y);
    EXPECT_EQ(cref, z);
}

TEST(Domain, noHalos)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    noHalos<unsigned, double>(rank, nRanks);
    noHalos<uint64_t, double>(rank, nRanks);
    noHalos<unsigned, float>(rank, nRanks);
    noHalos<uint64_t, float>(rank, nRanks);
}

/*! \brief a minimal initial domain.sync() test with halos
 *
 * The two global particles at 0.5^3 and 0.6^3 together with a tree
 * bucketSize of 1 is quite nasty, as it maxes out the tree division depth,
 * because the two particle always end up in the same node at all division levels.
 */
template<class I, class T>
void withHalos(int rank, int nRanks)
{
    int bucketSize = 1;
    Domain<I, T> domain(rank, nRanks, bucketSize);

    std::vector<T> x{0.5, 0.6};
    std::vector<T> y{0.5, 0.6};
    std::vector<T> z{0.5, 0.6};
    std::vector<T> h{0.2, 0.22}; // in range

    domain.sync(x,y,z,h);

    if (rank == 0)
    {
        EXPECT_EQ(domain.startIndex(), 0);
        EXPECT_EQ(domain.endIndex(), 2);
    }
    else if (rank == 1)
    {
        EXPECT_EQ(domain.startIndex(), 2);
        EXPECT_EQ(domain.endIndex(), 4);
    }

    std::vector<T> cref{0.5, 0.5, 0.6, 0.6};
    std::vector<T> href{0.2, 0.2, 0.22, 0.22};

    EXPECT_EQ(cref, x);
    EXPECT_EQ(cref, y);
    EXPECT_EQ(cref, z);
    EXPECT_EQ(href, h);
}

TEST(Domain, halos)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    withHalos<unsigned, double>(rank, nRanks);
    withHalos<uint64_t, double>(rank, nRanks);
    withHalos<unsigned, float>(rank, nRanks);
    withHalos<uint64_t, float>(rank, nRanks);
}


/*! \brief A test for one initial domain.sync() with halos
 *
 * Tests correct assignment and distribution of the global tree
 * plus halo exchange.
 */
template<class I, class T>
void moreHalos(int rank, int nRanks)
{
    int bucketSize = 4;
    Domain<I, T> domain(rank, nRanks, bucketSize);

    // node boundaries     |--(0,0)----|---------(0,7)-------------|-----(7,0)----------|-------(7,7)------|
    // indices             0    1      2      3      4      5      6      7      8      9      10     11
    std::vector<T> xGlobal{0.0, 0.11,  0.261, 0.281, 0.301, 0.321, 0.521, 0.541, 0.561, 0.761, 0.781, 1.000};
    std::vector<T> yGlobal{0.0, 0.12,  0.262, 0.282, 0.302, 0.322, 0.522, 0.542, 0.562, 0.762, 0.781, 1.000};
    std::vector<T> zGlobal{0.0, 0.13,  0.263, 0.283, 0.303, 0.323, 0.523, 0.543, 0.563, 0.763, 0.781, 1.000};
    std::vector<T> hGlobal{0.1, 0.101, 0.102, 0.103, 0.104, 0.105, 0.106, 0.107, 0.108, 0.109, 0.110, 0.111};
    // sync result rank 0: |---------assignment-------------------|------halos---------|
    // sync result rank 1:             |-----halos----------------|-----------assignment-------------------|

    std::vector<T> x,y,z,h;
    // rank 0 gets particles with even index before the sync
    // rank 1 gets particles with uneven index before the sync
    for(int i = rank; i < xGlobal.size(); i+=nRanks)
    {
        x.push_back(xGlobal[i]);
        y.push_back(yGlobal[i]);
        z.push_back(zGlobal[i]);
        h.push_back(hGlobal[i]);
    }

    domain.sync(x,y,z,h);

    if (rank == 0)
    {
        EXPECT_EQ(domain.startIndex(), 0);
        EXPECT_EQ(domain.endIndex(), 6);
        EXPECT_EQ(x.size(), 9);
        EXPECT_EQ(y.size(), 9);
        EXPECT_EQ(z.size(), 9);
        EXPECT_EQ(h.size(), 9);
    }
    else if (rank == 1)
    {
        EXPECT_EQ(domain.startIndex(), 4);
        EXPECT_EQ(domain.endIndex(), 10);
        EXPECT_EQ(x.size(), 10);
        EXPECT_EQ(y.size(), 10);
        EXPECT_EQ(z.size(), 10);
        EXPECT_EQ(h.size(), 10);
    }

    int gstart = (rank == 0) ? 0 : 2;
    int gend   = (rank == 0) ? 9 : 12;

    std::vector<T> xref{xGlobal.begin() + gstart, xGlobal.begin() + gend};
    std::vector<T> yref{yGlobal.begin() + gstart, yGlobal.begin() + gend};
    std::vector<T> zref{zGlobal.begin() + gstart, zGlobal.begin() + gend};
    std::vector<T> href{hGlobal.begin() + gstart, hGlobal.begin() + gend};

    EXPECT_EQ(x, xref);
    EXPECT_EQ(y, yref);
    EXPECT_EQ(z, zref);
    EXPECT_EQ(h, href);
}

TEST(Domain, moreHalos)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    moreHalos<unsigned, double>(rank, nRanks);
    moreHalos<uint64_t, double>(rank, nRanks);
    moreHalos<unsigned, float>(rank, nRanks);
    moreHalos<uint64_t, float>(rank, nRanks);
}

/*! \brief tests that after a domain.sync(), the previous halo exchange pattern can be reapplied
 *
 * E.g. when exchanging densities
 */
template<class I, class T>
void postSyncHalos(int rank, int nRanks)
{
    int bucketSize = 4;
    Domain<I, T> domain(rank, nRanks, bucketSize);

    // node boundaries     |--(0,0)----|---------(0,7)-------------|-----(7,0)----------|-------(7,7)------|
    // indices             0    1      2      3      4      5      6      7      8      9      10     11
    std::vector<T> xGlobal{0.0, 0.11,  0.261, 0.281, 0.301, 0.321, 0.521, 0.541, 0.561, 0.761, 0.781, 1.000};
    std::vector<T> yGlobal{0.0, 0.12,  0.262, 0.282, 0.302, 0.322, 0.522, 0.542, 0.562, 0.762, 0.781, 1.000};
    std::vector<T> zGlobal{0.0, 0.13,  0.263, 0.283, 0.303, 0.323, 0.523, 0.543, 0.563, 0.763, 0.781, 1.000};
    std::vector<T> hGlobal{0.1, 0.101, 0.102, 0.103, 0.104, 0.105, 0.106, 0.107, 0.108, 0.109, 0.110, 0.111};
    // sync result rank 0: |---------assignment-------------------|------halos---------|
    // sync result rank 1:             |-----halos----------------|-----------assignment-------------------|

    std::vector<T> x, y, z, h;
    // rank 0 gets particles with even index before the sync
    // rank 1 gets particles with uneven index before the sync
    for (int i = rank; i < xGlobal.size(); i += nRanks)
    {
        x.push_back(xGlobal[i]);
        y.push_back(yGlobal[i]);
        z.push_back(zGlobal[i]);
        h.push_back(hGlobal[i]);
    }

    domain.sync(x,y,z,h);

    std::vector<T> density, refDensity;
    if (rank == 0) {
        // before halo exchange
        density    = std::vector<T>{10,11,12,13,14,15, 0, 0, 0};
        // after
        refDensity = std::vector<T>{10,11,12,13,14,15,20,21,22};
    }
    else {
        // before halo exchange
        density    = std::vector<T>{0, 0, 0, 0, 20,21,22,23,24,25};
        // after
        refDensity = std::vector<T>{12,13,14,15,20,21,22,23,24,25};
    }

    domain.exchangeHalos(density);

    EXPECT_EQ(density, refDensity);
}

TEST(Domain, postSyncHalos)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    postSyncHalos<unsigned, double>(rank, nRanks);
    postSyncHalos<uint64_t, double>(rank, nRanks);
    postSyncHalos<unsigned, float>(rank, nRanks);
    postSyncHalos<uint64_t, float>(rank, nRanks);
}

/*! \brief Performs twice domain.sync(), with a particle coordinate update in between
 *
 * This ensures that the domain correctly remembers the array layout from the previous sync,
 * then correctly updates the global tree/assignment to reflect the changed coordinates.
 */
template<class I, class T>
void multiStepSync(int rank, int nRanks)
{
    int bucketSize = 4;
    Domain<I, T> domain(rank, nRanks, bucketSize);

    // node boundaries     |--(0,0)----|---------(0,7)-------------|-----(7,0)----------|-------(7,7)------|
    // indices             0    1      2      3      4      5      6      7      8      9      10     11
    std::vector<T> xGlobal{0.0, 0.11,  0.261, 0.281, 0.301, 0.321, 0.521, 0.541, 0.561, 0.761, 0.781, 1.000};
    std::vector<T> yGlobal{0.0, 0.12,  0.262, 0.282, 0.302, 0.322, 0.522, 0.542, 0.562, 0.762, 0.781, 1.000};
    std::vector<T> zGlobal{0.0, 0.13,  0.263, 0.283, 0.303, 0.323, 0.523, 0.543, 0.563, 0.763, 0.781, 1.000};
    std::vector<T> hGlobal{0.1, 0.101, 0.102, 0.103, 0.104, 0.105, 0.106, 0.107, 0.108, 0.109, 0.110, 0.111};
    // sync result rank 0: |---------assignment-------------------|------halos---------|
    // sync result rank 1:             |-----halos----------------|-----------assignment-------------------|

    std::vector<T> x, y, z, h;
    // rank 0 gets particles with even index before the sync
    // rank 1 gets particles with uneven index before the sync
    for (int i = rank; i < xGlobal.size(); i += nRanks)
    {
        x.push_back(xGlobal[i]);
        y.push_back(yGlobal[i]);
        z.push_back(zGlobal[i]);
        h.push_back(hGlobal[i]);
    }

    domain.sync(x,y,z,h);

    // now a particle on rank 0 gets moved into an area of the global tree that's on rank 1
    if (rank == 0)
    {
        x[1] = 0.811;
        y[1] = 0.812;
        z[1] = 0.813;
    }

    domain.sync(x,y,z,h);

    if (rank == 0)
    {
        //                 |--------assignment--------------|-------halos-------|
        std::vector<T> xref{0.0, 0.261, 0.281, 0.301, 0.321, 0.521, 0.541, 0.561};
        std::vector<T> yref{0.0, 0.262, 0.282, 0.302, 0.322, 0.522, 0.542, 0.562};
        std::vector<T> zref{0.0, 0.263, 0.283, 0.303, 0.323, 0.523, 0.543, 0.563};
        std::vector<T> href{0.1, 0.102, 0.103, 0.104, 0.105, 0.106, 0.107, 0.108};

        EXPECT_EQ(x, xref);
        EXPECT_EQ(y, yref);
        EXPECT_EQ(z, zref);
        EXPECT_EQ(h, href);
        EXPECT_EQ(domain.startIndex(), 0);
        EXPECT_EQ(domain.endIndex(), 5);
    }
    if (rank == 1)
    {   //                 |--------halos--------------|---------assignment----------------------------|
        std::vector<T> xref{0.261, 0.281, 0.301, 0.321, 0.521, 0.541, 0.561, 0.761, 0.781, 0.811, 1.000};
        std::vector<T> yref{0.262, 0.282, 0.302, 0.322, 0.522, 0.542, 0.562, 0.762, 0.781, 0.812, 1.000};
        std::vector<T> zref{0.263, 0.283, 0.303, 0.323, 0.523, 0.543, 0.563, 0.763, 0.781, 0.813, 1.000};
        std::vector<T> href{0.102, 0.103, 0.104, 0.105, 0.106, 0.107, 0.108, 0.109, 0.110, 0.101, 0.111};
        EXPECT_EQ(domain.startIndex(), 4);
        EXPECT_EQ(domain.endIndex(), 11);
    }
}

TEST(Domain, multiStepSync)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    multiStepSync<unsigned, double>(rank, nRanks);
    multiStepSync<uint64_t, double>(rank, nRanks);
    multiStepSync<unsigned, float>(rank, nRanks);
    multiStepSync<uint64_t, float>(rank, nRanks);
}
