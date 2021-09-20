
#include "gtest/gtest.h"

#include "ryoanji/buildtree.h"
#include "ryoanji/dataset.h"

TEST(Buildtree, bounds)
{
    constexpr int ncrit = 64;
    int numBodies = 1023;

    //! particles in [-3, 3]^3
    double extent = 3;
    auto bodies = makeCubeBodies(numBodies, extent);

    cudaVec<fvec4> bodyPos(numBodies, true);
    for (size_t i = 0; i < numBodies; ++i)
    {
        bodyPos[i] = bodies[i];
    }
    bodyPos.h2d();

    cudaVec<fvec4> bodyPos2(numBodies, true);

    Box box;
    cudaVec<int2> levelRange(32, true);
    cudaVec<CellData> sourceCells(numBodies);

    int3 counts = Build::tree<ncrit>(bodyPos, bodyPos2, box, levelRange, sourceCells);

    std::cout << "numLevels: " << counts.x << " numSources " << counts.y
              << " numLeaves " << counts.z << std::endl;

    std::cout << "box: " << box.X[0] << " " << box.X[1] << " " << box.X[2] << ", radius: " << box.R << std::endl;

    bodyPos.d2h();
    bodyPos2.d2h();
}
