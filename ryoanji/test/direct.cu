
#include "gtest/gtest.h"

#include "ryoanji/direct.cuh"

TEST(DirectSum, MatchCpu)
{
    int numBodies = 1023;
    int numTarget = std::min(512, numBodies); // Number of threads per block will be set to this value
    int numBlock  = std::min(128, (numBodies - 1) / numTarget + 1);

    int images = 0;
    float eps = 0.05;
    float cycle = 2 * M_PI;

    cudaVec<fvec4> bodyPos(numBodies, true);
    cudaVec<fvec4> bodyAcc(numBodies, true);

    directSum(numTarget, numBlock, images, eps, cycle, bodyPos, bodyAcc);

    bodyAcc.d2h();

    EXPECT_EQ(bodyAcc[0][0], 0.0);
}
