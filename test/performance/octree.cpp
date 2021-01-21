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
 * \brief Test morton code implementation
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <iostream>

#include "cstone/mortoncode.hpp"
#include "cstone/octree.hpp"
#include "cstone/octree_util.hpp"

#include "coord_samples/random.hpp"

using namespace cstone;

int main()
{
    using CodeType = unsigned;
    Box<double> box{-1, 1};

    int nParticles = 100000;
    int bucketSize = 10;

    RandomGaussianCoordinates<double, CodeType> randomBox(nParticles, box);

    // compute octree starting from default uniform octree
    auto [tree, counts] = computeOctree(randomBox.mortonCodes().data(),
                                        randomBox.mortonCodes().data() + nParticles,
                                        bucketSize);

    std::cout << "nNodes(tree): " << nNodes(tree) << std::endl;
}

