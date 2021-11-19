
#include "gtest/gtest.h"

#include "ryoanji/buildtree.h"
#include "ryoanji/grouptargets.h"
#include "ryoanji/dataset.h"

TEST(Buildtree, bounds)
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

    //! upload bodies to device
    bodyPos.h2d();

    cudaVec<fvec4> bodyPos2(numBodies, true);

    Box box;
    cudaVec<int2> levelRange(32, true);
    cudaVec<CellData> sourceCells(numBodies);

    int3 counts = Build::tree<ncrit>(bodyPos, bodyPos2, box, levelRange, sourceCells);

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
    sourceCells.d2h();

    float diameter = 2 * box.R / (1 << NBITS);
    fvec3 Xmin     = box.X - box.R;

    for (size_t i = 0; i < numBodies; ++i)
    {
        fvec3 iX = (make_fvec3(bodyPos[i]) - Xmin) / diameter;
        uint64_t key = getHilbert(make_int3(iX[0], iX[1], iX[2]));

        //std::cout << bodies[i] << " " << bodyPos[i] << " " << bodyPos2[i] << std::endl;
        //printf("%5f, %5f, %5f\n", bodies[i][0], bodies[i][1], bodies[i][2]);
        //printf("%5f, %5f, %5f, %021lo\n", bodyPos[i][0], bodyPos[i][1], bodyPos[i][2], key);
        //printf("%5f, %5f, %5f\n", bodyPos2[i][0], bodyPos2[i][1], bodyPos2[i][2]);
        //std::cout << bodies[i][j] << " " << bodyPos[i][j] << " " << bodyPos2[i][j] << std::endl;
        //std::cout << std::endl;
    }

}
