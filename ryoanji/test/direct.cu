
#include "gtest/gtest.h"

#include "cstone/gravity/treewalk.hpp"

#include "ryoanji/dataset.h"
#include "ryoanji/direct.cuh"

template<class T>
std::vector<fvec4> cpuReference(const std::vector<util::array<T, 4>>& bodies)
{
    size_t numBodies = bodies.size();

    std::vector<double> x(numBodies);
    std::vector<double> y(numBodies);
    std::vector<double> z(numBodies);
    std::vector<double> h(numBodies, 0.0);
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

    float G = 1.0;

    cstone::directSum(x.data(), y.data(), z.data(), h.data(), m.data(), numBodies, G,
                      ax.data(), ay.data(), az.data(), pot.data());

    std::vector<fvec4> acc(numBodies, fvec4{0, 0, 0, 0});

    for (size_t i = 0; i < numBodies; ++i)
    {
        acc[i] = fvec4{T(pot[i]), T(ax[i]), T(ay[i]), T(az[i])};
    }

    return acc;
}

TEST(DirectSum, MatchCpu)
{
    int npOnEdge  = 10;
    int numBodies = npOnEdge * npOnEdge * npOnEdge;

    // the CPU reference uses mass softening, while the GPU P2P kernel still uses plummer softening
    // so the only way to compare is without softening in both versions and make sure that
    // particles are not on top of each other
    float eps = 0.0;

    auto bodies = makeGridBodies(npOnEdge, 0.5);

    cudaVec<fvec4> bodyPos(numBodies, true);
    for (size_t i = 0; i < numBodies; ++i)
    {
        bodyPos[i] = bodies[i];
        //printf("%f %f %f %f\n", bodyPos[i][0], bodyPos[i][1], bodyPos[i][2], bodyPos[i][3]);
    }
    bodyPos.h2d();

    cudaVec<fvec4> bodyAcc(numBodies, true);
    bodyAcc.zeros();

    directSum(eps, bodyPos, bodyAcc);

    bodyAcc.d2h();

    auto refAcc = cpuReference(bodies);

    for (int i = 0; i < numBodies; ++i)
    {
        fvec3 ref   = {refAcc[i][1], refAcc[i][2], refAcc[i][3]};
        fvec3 probe = {bodyAcc[i][1], bodyAcc[i][2], bodyAcc[i][3]};

        EXPECT_NEAR(std::sqrt(norm2(ref - probe) / norm2(probe)), 0, 1e-6);
        // the potential
        EXPECT_NEAR(refAcc[i][0], bodyAcc[i][0], 1e-6);

        // printf("%f %f %f\n", ref[1], ref[2], ref[3]);
        // printf("%f %f %f\n", probe[1], probe[2], probe[3]);
    }
}
