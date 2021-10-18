
#include "gtest/gtest.h"

#include "cstone/gravity/treewalk.hpp"

#include "ryoanji/dataset.h"
#include "ryoanji/kernel.h"
#include "ryoanji/upwardpass.h"

//! @brief little P2M wrapper for the host without GPU textures
fvecP P2Mhost(int begin, int end, const fvec4* bodies, const fvec4& center)
{
    // the output multipole
    fvecP Mout(0);

    for (int i = begin; i < end; ++i)
    {
        // body and distance from center-mass
        fvec4 body = bodies[i];
        fvec3 dx   = make_fvec3(center - body);

        fvecP M;
        M[0] = body[3];
        Kernels<0,0,P-1>::P2M(M, dx);
        Mout += M;
    }

    return Mout;
}

TEST(Multipole, P2M)
{
    int numBodies = 1023;

    std::vector<fvec4> bodies = makeCubeBodies(numBodies);

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

    auto cstoneMultipole = cstone::particle2Multipole<double>(x.data(), y.data(), z.data(), m.data(), numBodies);

    fvec4 centerMass       = setCenter(0, numBodies, bodies.data());
    fvecP ryoanjiMultipole = P2Mhost(0, numBodies, bodies.data(), centerMass);

    EXPECT_NEAR(ryoanjiMultipole[0], cstoneMultipole.mass, 1e-6);

    EXPECT_NEAR(centerMass[0], cstoneMultipole.xcm , 1e-6);
    EXPECT_NEAR(centerMass[1], cstoneMultipole.ycm , 1e-6);
    EXPECT_NEAR(centerMass[2], cstoneMultipole.zcm , 1e-6);
    EXPECT_NEAR(centerMass[3], cstoneMultipole.mass, 1e-6);

    // the two multipoles are not directly comparable
    {
        //std::cout << "qxx " << cstoneMultipole.qxx << std::endl;
        //std::cout << "qxy " << cstoneMultipole.qxy << std::endl;
        //std::cout << "qxz " << cstoneMultipole.qxz << std::endl;
        //std::cout << "qyy " << cstoneMultipole.qyy << std::endl;
        //std::cout << "qyz " << cstoneMultipole.qyz << std::endl;
        //std::cout << "qzz " << cstoneMultipole.qzz << std::endl;
        //std::cout << std::endl;

        //for (int i = 0; i < NTERM; ++i)
        //{
        //    std::cout << i << " " << ryoanjiMultipole[i] << std::endl;
        //}
    }

    // compare M2P results on a test target
    {
        float eps2 = 0;
        fvec3 testTarget(-8, -8, -8);

        fvec4 acc(0, 0, 0, 0);
        acc = M2P(acc, testTarget, make_fvec3(centerMass), ryoanjiMultipole, eps2);
        //printf("test acceleration: %f %f %f %f\n", acc[0], acc[1], acc[2], acc[3]);

        // cstone is less precise
        //float ax = 0;
        //float ay = 0;
        //float az = 0;
        //cstone::multipole2particle(
        //    testTarget[0], testTarget[1], testTarget[2], cstoneMultipole, eps2, &ax, &ay, &az);
        //printf("cstone test acceleration: %f %f %f\n", ax, ay, az);

        double axd = 0;
        double ayd = 0;
        double azd = 0;
        double pot = 0;
        cstone::particle2particle(double(testTarget[0]), double(testTarget[1]), double(testTarget[2]), 0.0,
                                  x.data(), y.data(), z.data(), h.data(), m.data(), numBodies,
                                  &axd, &ayd, &azd, &pot);
        //printf("direct acceleration: %f %f %f\n", axd, ayd, azd);

        // compare ryoanji against the direct sum reference
        EXPECT_NEAR(acc[0], pot, 3e-5);
        EXPECT_NEAR(acc[1], axd, 1e-5);
        EXPECT_NEAR(acc[2], ayd, 1e-5);
        EXPECT_NEAR(acc[3], azd, 1e-5);
    }
}

