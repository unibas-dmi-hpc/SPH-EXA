
#include "gtest/gtest.h"

#include "ryoanji/buildtree.h"
#include "ryoanji/dataset.h"
#include "ryoanji/grouptargets.h"

bool isKeySorted(const cudaVec<fvec4>& bodies, const Box& box)
{
    const int numBodies = bodies.size();
    const int NBLOCK    = (numBodies - 1) / NTHREAD + 1;
    cudaVec<uint64_t> keys(numBodies, true);
    cudaVec<int>      ordering(numBodies);

    getKeys<<<NBLOCK, NTHREAD>>>(numBodies, box, bodies.d(), keys.d(), ordering.d());

    keys.d2h();
    return std::is_sorted(keys.h(), keys.h() + numBodies);
}

TEST(GroupTargets, construct)
{
    constexpr int ncrit = 64;
    int numBodies = 1024;

    //! particles in [-3, 3]^3
    float extent = 3;
    auto bodies = makeCubeBodies(numBodies, extent);

    cudaVec<fvec4> bodyPos(numBodies, true);
    for (int i = 0; i < numBodies; ++i)
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

    cudaVec<int2> targetRange(numBodies, true);
    Group group;
    int numTargets = group.targets(bodyPos, bodyPos2, box, targetRange, 1);
    fprintf(stdout, "num targets: %d\n", numTargets);

    //! download bodies and tree cells
    bodyPos.d2h();
    bodyPos2.d2h();
    sourceCellsLoc.d2h();
    levelRange.d2h();

    EXPECT_FALSE(isKeySorted(bodyPos, box));
    EXPECT_TRUE(isKeySorted(bodyPos2, box));

}
