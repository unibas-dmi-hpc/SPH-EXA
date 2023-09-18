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

#include "coord_samples/random.hpp"
#include "cstone/domain/assignment_gpu.cuh"
#include "cstone/domain/assignment.hpp"

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

    std::vector<KeyType> keys(numParticles);

    // Note: NOT sorted in SFC order
    std::vector<T> x(numParticles);
    std::vector<T> y(numParticles);
    std::vector<T> z(numParticles);
    initCoordinates(x, y, z, box, rank);

    int bucketSize = 20;

    GlobalAssignment<KeyType, T> assignment(rank, numRanks, bucketSize, box);
    BufferDescription bufDesc{0, numParticles, numParticles};
    std::vector<unsigned> sfcScratchCpu;
    SfcSorter<LocalIndex, std::vector<unsigned>> cpuGather(sfcScratchCpu);

    std::vector<T> s0, s1;
    LocalIndex exchangeSizeCpu =
        assignment.assign(bufDesc, cpuGather, s0, s1, keys.data(), x.data(), y.data(), z.data());

    thrust::device_vector<KeyType> d_keys;
    reallocateDevice(d_keys, numParticles, 1.0);

    thrust::device_vector<T> d_x = x;
    thrust::device_vector<T> d_y = y;
    thrust::device_vector<T> d_z = z;

    GlobalAssignmentGpu<KeyType, T> assignmentGpu(rank, numRanks, bucketSize, box);
    thrust::device_vector<unsigned> sfcScratch;
    GpuSfcSorter<LocalIndex, thrust::device_vector<unsigned>> deviceSort(sfcScratch);

    thrust::device_vector<T> d_s0, d_s1;
    bufDesc.size =
        assignmentGpu.assign(bufDesc, deviceSort, d_s0, d_s1, rawPtr(d_keys), rawPtr(d_x), rawPtr(d_y), rawPtr(d_z));

    ASSERT_EQ(exchangeSizeCpu, bufDesc.size);
    EXPECT_EQ(assignment.treeLeaves().size(), assignmentGpu.treeLeaves().size());

    reallocate(exchangeSizeCpu, keys, x, y, z);

    reallocateDevice(d_keys, bufDesc.size, 1.01);
    reallocateDevice(d_x, bufDesc.size, 1.01);
    reallocateDevice(d_y, bufDesc.size, 1.01);
    reallocateDevice(d_z, bufDesc.size, 1.01);

    std::vector<double> dummy;
    auto [exchangeStartCpu, cpuKeyView] =
        assignment.distribute(bufDesc, cpuGather, dummy, dummy, keys.data(), x.data(), y.data(), z.data());

    thrust::device_vector<T> sendScratch, receiveScratch;
    auto [exchangeStart, devKeyView] = assignmentGpu.distribute(bufDesc, deviceSort, sendScratch, receiveScratch,
                                                                rawPtr(d_keys), rawPtr(d_x), rawPtr(d_y), rawPtr(d_z));

    EXPECT_EQ(exchangeStart, exchangeStartCpu);
    EXPECT_EQ(devKeyView.size(), cpuKeyView.size());

    {
        std::vector<KeyType> keyDownload(devKeyView.size());
        thrust::copy_n(thrust::device_pointer_cast(devKeyView.data()), devKeyView.size(), keyDownload.data());
        EXPECT_TRUE(std::equal(keyDownload.begin(), keyDownload.end(), cpuKeyView.begin()));
    }
}

TEST(AssignmentGpu, matchTreeCpu)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    randomGaussianAssignment<uint64_t, double>(rank, numRanks);
}
