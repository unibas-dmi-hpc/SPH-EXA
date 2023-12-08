/*
 * MIT License
 *
 * Copyright (c) 2023 CSCS, ETH Zurich
 *               2023 University of Basel
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
 * @brief Unit tests for rasterization and power spectrum calculations
 *
 * @author Osman Seckin Simsek <osman.simsek@unibas.ch>
 */

#include <iostream>

#include "gtest/gtest.h"

#include "sph/mesh.hpp"

using namespace sphexa;

// Not really sure how to implement this test using MPI
TEST(GlobalMesh, makeGlobalMesh)
{
    int rank = 0, numRanks = 1;
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    int gridDim = 10;

    Mesh<double> mesh(rank, numRanks, gridDim);

    EXPECT_EQ(mesh.inbox_.size[0], 10);
    EXPECT_EQ(mesh.inbox_.size[1], 10);
    EXPECT_EQ(mesh.inbox_.size[2], 10);
}

TEST(GlobalMesh, fftFreq)
{
    // Test even variation
    int    n  = 10;
    double dt = 1.0 / n;

    std::vector<double> freq(n);

    fftfreq(freq, n, dt);

    EXPECT_EQ(freq[1], 1);
    EXPECT_EQ(freq[5], -5);
    EXPECT_EQ(freq[9], -1);

    // Test odd variation
    n  = 5;
    dt = 1.0 / n;

    fftfreq(freq, n, dt);

    EXPECT_EQ(freq[1], 1);
    EXPECT_EQ(freq[2], -2);
    EXPECT_EQ(freq[3], -1);
    EXPECT_EQ(freq[4], 0);
}