
#include "gtest/gtest.h"

#include "ryoanji/buildtree.h"
#include "ryoanji/dataset.h"
#include "ryoanji/grouptargets.h"
#include "ryoanji/upwardpass.h"

TEST(Buildtree, upsweep)
{
    constexpr int ncrit = 64;
    int numBodies = 1023;

    //! particles in [-3, 3]^3
    float extent = 3;
    auto bodies = makeCubeBodies(numBodies, extent);

    // set non-random corners
    bodies[0][0] = -extent;
    bodies[0][1] = -extent;
    bodies[0][2] = -extent;

    bodies[numBodies - 1][0] = extent;
    bodies[numBodies - 1][1] = extent;
    bodies[numBodies - 1][2] = extent;

    cudaVec<fvec4> bodyPos(numBodies, true);
    for (size_t i = 0; i < numBodies; ++i)
    {
        bodyPos[i] = bodies[i];
    }

    float totalMass = 0;
    for (size_t i = 0; i < numBodies; ++i)
    {
        totalMass += bodies[i][3];
    }
    std::cout << "totalMass " << totalMass << std::endl;

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

    cudaVec<int2> targetRange(numBodies, true);
    Group group;
    int numTargets = group.targets(bodyPos, bodyPos2, box, targetRange, 1);
    fprintf(stdout, "num targets: %d\n", numTargets);

    //! tree depth
    EXPECT_EQ(counts.x, 2);
    //! total tree cell count
    EXPECT_EQ(counts.y, 72);
    //! leaf cell count
    EXPECT_EQ(counts.z, 64);

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

    for (size_t i = 0; i < numSources; ++i)
    {
        float cellMass = 0;
        for (size_t j = sourceCellsLoc[i].body(); j < sourceCellsLoc[i].body() + sourceCellsLoc[i].nbody(); ++j)
        {
            cellMass += bodyPos[j][3];
        }

        // each multipole should have the total mass of referenced bodies in the first entry
        EXPECT_NEAR(cellMass, Multipole[i*NVEC4][0], 1e-5);
    }
}

