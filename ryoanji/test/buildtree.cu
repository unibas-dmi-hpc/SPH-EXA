
#include "gtest/gtest.h"

#include "cstone/sfc/common.hpp"

#include "ryoanji/buildtree.h"
#include "ryoanji/dataset.h"
#include "ryoanji/grouptargets.h"
#include "ryoanji/upwardpass.h"


uint64_t calcKey(const fvec3& pos, const Box& box)
{
    float diameter = 2 * box.R / (1 << NBITS);

    const fvec3 Xmin = box.X - box.R;
    const fvec3 iX   = (pos - Xmin) / diameter;
    return getHilbert(make_int3(iX[0], iX[1], iX[2]));
}

TEST(Buildtree, upsweep)
{
    constexpr int ncrit = 64;
    int numBodies = 8191;

    //! particles in [-3, 3]^3
    float extent = 3;
    auto bodies = makeCubeBodies(numBodies, extent);

    cudaVec<fvec4> bodyPos(numBodies, true);
    for (int i = 0; i < numBodies; ++i)
    {
        bodyPos[i] = bodies[i];
    }

    //! upload bodies to device
    bodyPos.h2d();

    cudaVec<fvec4> bodyPos2(numBodies, true);

    Box box;
    cudaVec<int2> levelRange(32, true);
    cudaVec<CellData> sourceCellsLoc(numBodies, true);

    int3 counts = Build::tree<ncrit>(bodyPos, bodyPos2, box, levelRange, sourceCellsLoc);

    int numLevels  = counts.x;
    int numSources = counts.y;
    int numLeafs   = counts.z;
    cudaVec<fvec4> sourceCenter(numSources, true);
    cudaVec<fvec4> Multipole(NVEC4 * numSources, true);

    float theta = 0.5;
    Pass::upward(numLeafs, numLevels, theta, levelRange, bodyPos, sourceCellsLoc, sourceCenter, Multipole);

    if (numBodies <= 1024)
    {
        //! tree depth
        EXPECT_EQ(counts.x, 2);
        //! total tree cell count
        EXPECT_EQ(counts.y, 72);
        //! leaf cell count
        EXPECT_EQ(counts.z, 64);
    }

    //! check that the correct bounding box was calculated
    Box refBox{{0.0f, 0.0f, 0.0f}, extent*1.1f};
    EXPECT_EQ(refBox.X[0], box.X[0]);
    EXPECT_EQ(refBox.X[1], box.X[1]);
    EXPECT_EQ(refBox.X[2], box.X[2]);
    EXPECT_EQ(refBox.R, box.R);

    //! download bodies and tree cells
    bodyPos.d2h();
    bodyPos2.d2h();
    sourceCellsLoc.d2h();
    levelRange.d2h();

    sourceCenter.d2h();
    Multipole.d2h();

    std::vector<uint64_t> level2keys(64);
    for (int i = levelRange[2].x; i < levelRange[2].y; ++i)
    {
        uint64_t key = calcKey(make_fvec3(sourceCenter[i]), box);
        key = cstone::enclosingBoxCode(key, 2);
        level2keys[i - levelRange[2].x] = key;
    }
    // level 2 cell keys should be unique
    EXPECT_TRUE(std::unique(level2keys.begin(), level2keys.end()) == level2keys.end());

    std::vector<int> bodyIndexed(numBodies, 0);
    for (size_t i = levelRange[2].x; i < levelRange[2].y; ++i)
    {
        EXPECT_EQ(levelRange[2].y - levelRange[2].x, 64);
        for (size_t j = sourceCellsLoc[i].body(); j < sourceCellsLoc[i].body() + sourceCellsLoc[i].nbody(); ++j)
        {
            bodyIndexed[j] += 1;
        }
    }
    // each body should be referenced exactly once by all level-2 nodes together
    EXPECT_EQ(std::count(begin(bodyIndexed), end(bodyIndexed), 1), numBodies);

    std::cout << "num cells " << counts.y << std::endl;
    for (int i = 0; i < numSources; ++i)
    {
        for (int d = 0; d < 3; ++d)
        {
            EXPECT_GT(sourceCenter[i][d], box.X[d] - box.R);
            EXPECT_LT(sourceCenter[i][d], box.X[d] + box.R);
        }
        EXPECT_TRUE(sourceCenter[i][3] < 4 * box.R * box.R);

        uint64_t cellKey = cstone::enclosingBoxCode(calcKey(make_fvec3(sourceCenter[i]), box), sourceCellsLoc[i].level());

        float cellMass = 0;
        for (int j = sourceCellsLoc[i].body(); j < sourceCellsLoc[i].body() + sourceCellsLoc[i].nbody(); ++j)
        {
            cellMass += bodyPos[j][3];

            uint64_t bodyKey = calcKey(make_fvec3(bodyPos[j]), box);
            // check that the referenced body really lies inside the cell
            // this is true if the keys, truncated to the first key-in-cell match
            EXPECT_EQ(cellKey, cstone::enclosingBoxCode(bodyKey, sourceCellsLoc[i].level()));
        }

        // each multipole should have the total mass of referenced bodies in the first entry
        EXPECT_NEAR(cellMass, Multipole[i*NVEC4][0], 1e-5);
    }
}

