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

#include "cstone/sfc/sfc.hpp"

#include "dataset.hpp"

#include "ryoanji/gpu_config.h"
#include "ryoanji/treebuilder.cuh"
#include "ryoanji/types.h"
#include "ryoanji/upwardpass.cuh"

using namespace ryoanji;

void checkBodyIndexing(int numBodies, CellData* tree, int numSources)
{
    std::vector<int> bodyIndexed(numBodies, 0);
    for (size_t i = 1; i < numSources; ++i) // we don't check the root
    {
        if (tree[i].isLeaf())
        {
            for (size_t j = tree[i].body(); j < tree[i].body() + tree[i].nbody(); ++j)
            {
                bodyIndexed[j]++;
            }
        }
        else
        {
            EXPECT_EQ(tree[i].nchild(), 8);
            EXPECT_TRUE(tree[i].nbody() > 0);
        }

        if (tree[i].level() > 1)
        {
            int parent = tree[i].parent();
            EXPECT_LE(tree[parent].child(), i);
            EXPECT_LT(i, tree[parent].child() + 8);
        }
    }

    // each body should be referenced exactly once by all leaves put together
    EXPECT_EQ(std::count(begin(bodyIndexed), end(bodyIndexed), 1), numBodies);
}

template<class T, class MType>
void checkUpsweep(const cstone::Box<T>& box, const thrust::host_vector<CellData>& sources,
                  const thrust::host_vector<Vec4<T>>& sourceCenter, const thrust::host_vector<Vec4<T>>& h_bodies,
                  const thrust::host_vector<MType>& Multipole)
{
    TreeNodeIndex numSources = sources.size();
    // the root is not set by the upsweep, so start from 1
    for (TreeNodeIndex i = 1; i < numSources; ++i)
    {
        if (sources[i].nbody())
        {
            EXPECT_GT(sourceCenter[i][0], box.xmin());
            EXPECT_LT(sourceCenter[i][0], box.xmax());
            EXPECT_GT(sourceCenter[i][1], box.xmin());
            EXPECT_LT(sourceCenter[i][1], box.xmax());
            EXPECT_GT(sourceCenter[i][2], box.xmin());
            EXPECT_LT(sourceCenter[i][2], box.xmax());

            EXPECT_TRUE(sourceCenter[i][3] < box.maxExtent() * box.maxExtent());

            uint64_t cellKey =
                cstone::enclosingBoxCode(cstone::sfc3D<cstone::SfcKind<uint64_t>>(
                                             sourceCenter[i][0], sourceCenter[i][1], sourceCenter[i][2], box),
                                         sources[i].level());

            T cellMass = 0;
            for (int j = sources[i].body(); j < sources[i].body() + sources[i].nbody(); ++j)
            {
                cellMass += h_bodies[j][3];

                uint64_t bodyKey =
                    cstone::sfc3D<cstone::SfcKind<uint64_t>>(h_bodies[j][0], h_bodies[j][1], h_bodies[j][2], box);
                // check that the referenced body really lies inside the cell
                // this is true if the keys, truncated to the first key-in-cell match
                EXPECT_EQ(cellKey, cstone::enclosingBoxCode(bodyKey, sources[i].level()));
            }

            // each multipole should have the total mass of referenced bodies in the first entry
            EXPECT_NEAR(cellMass, Multipole[i][0], 1e-5);
        }
        else
        {
            EXPECT_EQ(sourceCenter[i][0], 0.0f);
            EXPECT_EQ(sourceCenter[i][1], 0.0f);
            EXPECT_EQ(sourceCenter[i][2], 0.0f);
        }
    }
}

TEST(Buildtree, cstone)
{
    using T             = float;
    using MultipoleType = SphericalMultipole<T, 4>;

    int numBodies = (1 << 16) - 1;
    T   extent    = 3;
    T   theta     = 0.75;

    thrust::host_vector<Vec4<T>> h_bodies(numBodies);
    makeCubeBodies(h_bodies.data(), numBodies, extent);
    // upload to device
    thrust::device_vector<Vec4<T>> bodyPos = h_bodies;

    cstone::Box<T> box(-extent * 1.1, extent * 1.1);

    TreeBuilder<uint64_t> treeBuilder;
    int                   numSources = treeBuilder.update(rawPtr(bodyPos.data()), numBodies, box);

    thrust::device_vector<CellData> sources(numSources);
    std::vector<int2>               levelRange(cstone::maxTreeLevel<uint64_t>{} + 1);

    int highestLevel = treeBuilder.extract(rawPtr(sources.data()), levelRange.data());
    // download from device
    h_bodies = bodyPos;

    thrust::device_vector<Vec4<T>>       sourceCenter(numSources);
    thrust::device_vector<MultipoleType> Multipole(numSources);

    ryoanji::upsweep(numSources,
                     highestLevel,
                     theta,
                     levelRange.data(),
                     rawPtr(bodyPos.data()),
                     rawPtr(sources.data()),
                     rawPtr(sourceCenter.data()),
                     rawPtr(Multipole.data()));

    thrust::host_vector<CellData> h_sources      = sources;
    thrust::host_vector<Vec4<T>>  h_sourceCenter = sourceCenter;

    thrust::host_vector<MultipoleType> h_Multipole = Multipole;

    checkBodyIndexing(numBodies, h_sources.data(), numSources);
    checkUpsweep(box, h_sources, h_sourceCenter, h_bodies, h_Multipole);
}
