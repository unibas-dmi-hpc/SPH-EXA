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
 * @brief  Tests the Ryoanji-converted tree and upsweep
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "ryoanji/buildtree_cs.hpp"
#include "ryoanji/dataset.h"
#include "ryoanji/types.h"
#include "ryoanji/upwardpass.h"


void checkBodyIndexing(int numBodies, const std::vector<CellData>& tree)
{
    std::vector<int> bodyIndexed(numBodies, 0);
    for (size_t i = 0; i < tree.size(); ++i)
    {
        if (tree[i].isLeaf())
        {
            for (size_t j = tree[i].body(); j < tree[i].body() + tree[i].nbody(); ++j)
            {
                bodyIndexed[j]++;
            }
        }
    }

    // each body should be referenced exactly once by all leaves put together
    EXPECT_EQ(std::count(begin(bodyIndexed), end(bodyIndexed), 1), numBodies);
}

void checkUpsweep(const Box& box, const cudaVec<CellData>& sources, const cudaVec<fvec4>& sourceCenter,
                  const cudaVec<fvec4>& bodyPos, const cudaVec<fvec4>& Multipole)
{
    cstone::Box<float> csBox(box.X[0] - box.R, box.X[0] + box.R,
                             box.X[1] - box.R, box.X[1] + box.R,
                             box.X[2] - box.R, box.X[2] + box.R,
                             false, false, false);

    int numSources = sources.size();
    // the root is not set by the upsweep, so start from 1
    for (int i = 1; i < numSources; ++i)
    {
        for (int d = 0; d < 3; ++d)
        {
            EXPECT_GT(sourceCenter[i][d], box.X[d] - box.R);
            EXPECT_LT(sourceCenter[i][d], box.X[d] + box.R);
        }
        EXPECT_TRUE(sourceCenter[i][3] < 4 * box.R * box.R);

        uint64_t cellKey = cstone::enclosingBoxCode(
            cstone::sfc3D<cstone::SfcKind<uint64_t>>(sourceCenter[i][0], sourceCenter[i][1], sourceCenter[i][2], csBox),
            sources[i].level());

        float cellMass = 0;
        for (int j = sources[i].body(); j < sources[i].body() + sources[i].nbody(); ++j)
        {
            cellMass += bodyPos[j][3];

            uint64_t bodyKey =
                cstone::sfc3D<cstone::SfcKind<uint64_t>>(bodyPos[j][0], bodyPos[j][1], bodyPos[j][2], csBox);
            // check that the referenced body really lies inside the cell
            // this is true if the keys, truncated to the first key-in-cell match
            EXPECT_EQ(cellKey, cstone::enclosingBoxCode(bodyKey, sources[i].level()));
        }

        // each multipole should have the total mass of referenced bodies in the first entry
        EXPECT_NEAR(cellMass, Multipole[i * NVEC4][0], 1e-5);
    }
}

TEST(Buildtree, cstone)
{
    int numBodies = 8191;
    float extent = 3;
    float theta = 0.75;

    auto bodies = makeCubeBodies(numBodies, extent);

    Box box{ {0.0f}, extent * 1.1f };

    auto [highestLevel, tree, levelRangeCs] = buildFromCstone(bodies, box);
    checkBodyIndexing(numBodies, tree);

    int numSources = tree.size();

    cudaVec<fvec4> bodyPos(numBodies, true);
    std::copy(bodies.begin(), bodies.end(), bodyPos.h());
    bodyPos.h2d();

    cudaVec<int2> levelRange(levelRangeCs.size(), true);
    std::copy(levelRangeCs.begin(), levelRangeCs.end(), levelRange.h());
    levelRange.h2d();

    cudaVec<CellData> sources(numSources, true);
    std::copy(tree.begin(), tree.end(), sources.h());
    sources.h2d();

    cudaVec<fvec4> sourceCenter(numSources, true);
    cudaVec<fvec4> Multipole(NVEC4 * numSources, true);

    int numLeaves = -1;
    Pass::upward(numLeaves, highestLevel, theta, levelRange, bodyPos, sources, sourceCenter, Multipole);
    sourceCenter.d2h();
    Multipole.d2h();

    checkUpsweep(box, sources, sourceCenter, bodyPos, Multipole);
}
