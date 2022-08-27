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
 * @brief Halo exchange test
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <cassert>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/halos/exchange_halos_gpu.cuh"

using namespace cstone;

void gpuDirect(int rank)
{
    std::vector<int> msg{0, 1, 2, 3, 4};
    thrust::device_vector<int> src  = msg;
    thrust::device_vector<int> dest = std::vector<int>{-1, -1, -1, -1, -1};

    std::vector<MPI_Request> sendRequests;
    int tag = 0;

    if (rank == 0)
    {
        [[maybe_unused]] int err = MPI_Send(rawPtr(src), msg.size(), MPI_INT, 1, tag, MPI_COMM_WORLD);
        assert(err == MPI_SUCCESS);
    }
    else
    {
        [[maybe_unused]] int err = mpiRecvSync(rawPtr(dest), msg.size(), 0, tag, MPI_STATUS_IGNORE);
        assert(err == MPI_SUCCESS);

        std::vector<int> probe(msg.size());
        thrust::copy(dest.begin(), dest.end(), probe.begin());
        EXPECT_EQ(probe, msg);
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

#ifdef USE_GPU_DIRECT
TEST(HaloExchange, gpuDirect)
#else
TEST(HaloExchange, DISABLED_gpuDirect)
#endif
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    constexpr int thisExampleRanks = 2;

    if (nRanks != thisExampleRanks) throw std::runtime_error("this test needs 2 ranks\n");

    gpuDirect(rank);
}

TEST(HaloExchange, gatherRanges)
{
    // list of marked halo cells/ranges
    std::vector<int> seq(30);
    std::iota(seq.begin(), seq.end(), 0);
    thrust::device_vector<int> src = seq;

    thrust::device_vector<unsigned> rangeScan    = std::vector<unsigned>{0, 4, 7};
    thrust::device_vector<unsigned> rangeOffsets = std::vector<unsigned>{4, 12, 22};
    int totalCount                               = 10;

    thrust::device_vector<int> buffer = std::vector<int>(totalCount);

    gatherRanges(rawPtr(rangeScan), rawPtr(rangeOffsets), rangeScan.size(), rawPtr(src), rawPtr(buffer), totalCount);

    thrust::host_vector<int> h_buffer = buffer;
    thrust::host_vector<int> ref      = std::vector<int>{4, 5, 6, 7, 12, 13, 14, 22, 23, 24};

    EXPECT_EQ(h_buffer, ref);
}

void simpleTest(int thisRank)
{
    int nRanks = 2;
    std::vector<int> nodeList{0, 1, 10, 11};
    std::vector<int> offsets{0, 1, 3, 6, 10};

    int localCount, localOffset;
    if (thisRank == 0)
    {
        localCount  = 3;
        localOffset = 0;
    }
    if (thisRank == 1)
    {
        localCount  = 7;
        localOffset = 3;
    }

    SendList incomingHalos(nRanks);
    SendList outgoingHalos(nRanks);

    if (thisRank == 0)
    {
        incomingHalos[1].addRange(3, 6);
        incomingHalos[1].addRange(6, 10);
        outgoingHalos[1].addRange(0, 1);
        outgoingHalos[1].addRange(1, 3);
    }
    if (thisRank == 1)
    {
        incomingHalos[0].addRange(0, 1);
        incomingHalos[0].addRange(1, 3);
        outgoingHalos[0].addRange(3, 6);
        outgoingHalos[0].addRange(6, 10);
    }

    std::vector<double> x(*offsets.rbegin());
    std::vector<float> y(*offsets.rbegin());
    std::vector<util::array<int, 3>> velocity(*offsets.rbegin());

    int xshift = 20;
    int yshift = 30;
    for (int i = 0; i < localCount; ++i)
    {
        int eoff       = localOffset + i;
        x[eoff]        = eoff + xshift;
        y[eoff]        = eoff + yshift;
        velocity[eoff] = util::array<int, 3>{eoff, eoff + 1, eoff + 2};
    }

    if (thisRank == 0)
    {
        std::vector<double> xOrig{20, 21, 22, 0, 0, 0, 0, 0, 0, 0};
        std::vector<float> yOrig{30, 31, 32, 0, 0, 0, 0, 0, 0, 0};
        EXPECT_EQ(xOrig, x);
        EXPECT_EQ(yOrig, y);
    }
    if (thisRank == 1)
    {
        std::vector<double> xOrig{0, 0, 0, 23, 24, 25, 26, 27, 28, 29};
        std::vector<float> yOrig{0, 0, 0, 33, 34, 35, 36, 37, 38, 39};
        EXPECT_EQ(xOrig, x);
        EXPECT_EQ(yOrig, y);
    }

    thrust::device_vector<double> d_x                     = x;
    thrust::device_vector<float> d_y                      = y;
    thrust::device_vector<util::array<int, 3>> d_velocity = velocity;

    thrust::device_vector<char> sendBuffer    = std::vector<char>(7 * 24);
    thrust::device_vector<char> receiveBuffer = std::vector<char>(7 * 24);

    haloExchangeGpu(0, incomingHalos, outgoingHalos, sendBuffer, receiveBuffer, thrust::raw_pointer_cast(d_x.data()),
                    thrust::raw_pointer_cast(d_y.data()), thrust::raw_pointer_cast(d_velocity.data()));

    // download from device
    thrust::copy(d_x.begin(), d_x.end(), x.begin());
    thrust::copy(d_y.begin(), d_y.end(), y.begin());
    thrust::copy(d_velocity.begin(), d_velocity.end(), velocity.begin());

    std::vector<double> xRef{20, 21, 22, 23, 24, 25, 26, 27, 28, 29};
    std::vector<float> yRef{30, 31, 32, 33, 34, 35, 36, 37, 38, 39};
    std::vector<util::array<int, 3>> velocityRef{{0, 1, 2}, {1, 2, 3}, {2, 3, 4}, {3, 4, 5},  {4, 5, 6},
                                                 {5, 6, 7}, {6, 7, 8}, {7, 8, 9}, {8, 9, 10}, {9, 10, 11}};
    EXPECT_EQ(xRef, x);
    EXPECT_EQ(yRef, y);
    EXPECT_EQ(velocityRef, velocity);
}

TEST(HaloExchange, simpleTest)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    constexpr int thisExampleRanks = 2;

    if (nRanks != thisExampleRanks) throw std::runtime_error("this test needs 2 ranks\n");

    simpleTest(rank);
}
