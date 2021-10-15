
#include "gtest/gtest.h"

#include "cstone/gravity/treewalk.hpp"

#include "ryoanji/dataset.h"
#include "ryoanji/direct.cuh"

std::vector<fvec4> cpuReference(const std::vector<fvec4>& bodies, double eps)
{
    size_t numBodies = bodies.size();

    std::vector<double> x(numBodies);
    std::vector<double> y(numBodies);
    std::vector<double> z(numBodies);
    std::vector<double> m(numBodies);

    for (size_t i = 0; i < numBodies; ++i)
    {
        x[i] = bodies[i][0];
        y[i] = bodies[i][1];
        z[i] = bodies[i][2];
        m[i] = bodies[i][3];
    }

    std::vector<double> ax(numBodies);
    std::vector<double> ay(numBodies);
    std::vector<double> az(numBodies);
    std::vector<double> pot(numBodies);

    cstone::directSum(x.data(), y.data(), z.data(), m.data(), numBodies, eps*eps, ax.data(), ay.data(), az.data(),
                      pot.data());

    std::vector<fvec4> acc(numBodies, fvec4(0));

    for (size_t i = 0; i < numBodies; ++i)
    {
        acc[i] = fvec4(pot[i], ax[i], ay[i], az[i]);
    }

    return acc;
}

TEST(DirectSum, MatchCpu)
{
    int numBodies = 1023;
    float eps     = 0.05;

    auto bodies = makeCubeBodies(numBodies);

    cudaVec<fvec4> bodyPos(numBodies, true);
    for (size_t i = 0; i < numBodies; ++i)
    {
        bodyPos[i] = bodies[i];
    }
    bodyPos.h2d();

    cudaVec<fvec4> bodyAcc(numBodies, true);
    bodyAcc.zeros();

    directSum(eps, bodyPos, bodyAcc);

    bodyAcc.d2h();

    auto refAcc = cpuReference(bodies, eps);

    for (int i = 0; i < numBodies; ++i)
    {
        fvec3 ref   = {refAcc[i][1], refAcc[i][2], refAcc[i][3]};
        fvec3 probe = {bodyAcc[i][1], bodyAcc[i][2], bodyAcc[i][3]};

        EXPECT_TRUE(std::sqrt(norm(ref-probe)/norm(probe)) < 1e-6);
        // the potential
        EXPECT_NEAR(refAcc[i][0], bodyAcc[i][0], 1e-6);

        //printf("%f %f %f\n", ref[1], ref[2], ref[3]);
        //printf("%f %f %f\n", probe[1], probe[2], probe[3]);
    }
}
