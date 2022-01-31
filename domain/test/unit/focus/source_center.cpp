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
 * @brief Test source (mass) center computation
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "cstone/focus/source_center.hpp"
#include "cstone/tree/octree_util.hpp"

#include "coord_samples/random.hpp"

namespace cstone
{

template<class KeyType>
static void computeSourceCenter()
{
    LocalIndex numParticles = 20000;
    Box<double> box{-1, 1};
    unsigned csBucketSize = 16;

    RandomGaussianCoordinates<double, SfcKind<KeyType>> coords(numParticles, box);

    auto [csTree, csCounts] =
        computeOctree(coords.particleKeys().data(), coords.particleKeys().data() + numParticles, csBucketSize);
    Octree<KeyType> octree;
    octree.update(csTree.data(), nNodes(csTree));

    std::vector<double> masses(numParticles);
    std::generate(begin(masses), end(masses), [numParticles](){ return drand48() / numParticles; });
    std::vector<util::array<double, 4>> centers(octree.numTreeNodes());

    computeLeafMassCenter<double, double, double, KeyType>(coords.x(), coords.y(), coords.z(), masses,
                                                           coords.particleKeys(), octree, centers);
    upsweepMassCenter<double, KeyType>(octree, centers);

    util::array<double, 4> refRootCenter =
        massCenter<double, double, double>(coords.x(), coords.y(), coords.z(), masses, 0, numParticles);

    TreeNodeIndex rootNode = octree.levelOffset(0);
    std::cout << centers[rootNode][3] << std::endl;
    std::cout << refRootCenter[3] << std::endl;

    EXPECT_NEAR(centers[rootNode][3], refRootCenter[3], 1e-8);
}

TEST(FocusedOctree, sourceCenter)
{
    computeSourceCenter<unsigned>();
    computeSourceCenter<uint64_t>();
}

} // namespace cstone