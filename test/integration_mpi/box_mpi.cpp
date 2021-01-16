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
 * \brief Tests the global bounding box
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include <numeric>
#include <vector>

#include "cstone/box_mpi.hpp"

using namespace sphexa;

template<class T>
void globalMin(int rank, int nRanks)
{
    int nElements = 1000;
    std::vector<T> x(nElements);
    std::iota(begin(x), end(x), 1);
    for (auto& val : x)
        val /= (rank+1);

    T gmin = globalMin(begin(x), end(x));

    EXPECT_EQ(gmin, T(1)/nRanks);
}

TEST(GlobalBox, globalMin)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    globalMin<float>(rank, nRanks);
    globalMin<double>(rank, nRanks);
}

template<class T>
void globalMax(int rank, int nRanks)
{
    int nElements = 1000;
    std::vector<T> x(nElements);
    std::iota(begin(x), end(x), 1);
    for (auto& val : x)
        val /= (rank+1);

    T gmax = globalMax(begin(x), end(x));

    EXPECT_EQ(gmax, T(nElements));
}

TEST(GlobalBox, globalMax)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    globalMax<float>(rank, nRanks);
    globalMax<double>(rank, nRanks);
}

template<class T>
void makeGlobalBox(int rank, int nRanks)
{
    int nElements = 10;
    std::vector<T> x(nElements);
    std::iota(begin(x), end(x), 1);

    std::vector<T> y = x;
    std::vector<T> z = x;

    for (auto& val : x)
        val *= (rank+1);

    for (auto& val : y)
        val *= (rank+2);

    for (auto& val : z)
        val *= (rank+3);

    Box<T> box = makeGlobalBox(begin(x), end(x), begin(y), begin(z), true, true, true);

    Box<T> refBox{1, T(nElements*nRanks), 2, T(nElements*(nRanks+1)), 3, T(nElements*(nRanks+2)), true, true, true};

    EXPECT_EQ(box, refBox);
}

TEST(GlobalBox, makeGlobalBox)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    makeGlobalBox<float>(rank, nRanks);
    makeGlobalBox<double>(rank, nRanks);
}
