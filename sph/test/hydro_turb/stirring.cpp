
#include <vector>
#include <iostream>

#include "gtest/gtest.h"

#include "sph/hydro_turb/stirring.hpp"
#include "sph/hydro_turb/phases.hpp"

//! @brief test determination of numModes
TEST(Turbulence, stirParticle)
{
    /* test stirParticle
     *
     * 1. define an (x,y,z) particle
     * 2. create a few modes, amplitudes and phases
     * 3. apply stirring and check resulting accelerations
     *
     * choose your modes, phases and amplitudes wisely. use synthetic values that have a well-defined
     * effect on the resulting accelerations that's easy to reason about.
     *
     * Examples:
     * -if there is only one mode, the accelerations are proportional to the amplitudes
     * -the acceleration sum of two separate calls with modes m1 and m2 should be the same
     *  as one call with both modes
     *
     */
    using Tc = double;
    using T = double;
    using Ta = double;

    T tol = 1.e-8;
    size_t ndim = 3;

    Tc xi = 0.5;
    Tc yi = 0.5;
    Tc zi = 0.5;

    int numModes = 1;
    T modes[3] ={12.566370614359172,0,0}; // 2*k = 4*pi
    T phaseReal[3] = {1.0,1.0,1.0};
    T phaseImag[3] = {1.0,1.0,1.0};
    T amplitudes[1] = {2.0};


    auto [turbAx, turbAy, turbAz] = sph::stirParticle<Tc, Ta, T>(ndim, xi, yi, zi, numModes, modes, phaseReal, phaseImag, amplitudes);


    EXPECT_NEAR(turbAx,   1.9999996503088493, tol); //analytic value is 2.0, fortran code somehow is less accurate 
    EXPECT_NEAR(turbAy,  1.9999996503088493, tol);
    EXPECT_NEAR(turbAz,   1.9999996503088493, tol);

}

TEST(Turbulence, stirParticle2)
{
    /* test stirParticle
     *
     * 1. define an (x,y,z) particle
     * 2. create a few modes, amplitudes and phases
     * 3. apply stirring and check resulting accelerations
     *
     * choose your modes, phases and amplitudes wisely. use synthetic values that have a well-defined
     * effect on the resulting accelerations that's easy to reason about.
     *
     * Examples:
     * -if there is only one mode, the accelerations are proportional to the amplitudes
     * -the acceleration sum of two separate calls with modes m1 and m2 should be the same
     *  as one call with both modes
     *
     */
    using Tc = double;
    using T = double;
    using Ta = double;

    T tol = 1.e-8;
    size_t ndim = 3;

    Tc xi = -0.2;
    Tc yi = 0.3;
    Tc zi = -0.4;

    int numModes = 1;
    T modes[3] ={10.0,15.0,14.0};
    T phaseReal[3] = {1.1,1.2,1.3};
    T phaseImag[3] = {0.7,0.8,0.9};
    T amplitudes[1] = {2.0};


    auto [turbAx, turbAy, turbAz] = sph::stirParticle<Tc, Ta, T>(ndim, xi, yi, zi, numModes, modes, phaseReal, phaseImag, amplitudes);

    EXPECT_NEAR(turbAx, -2.1398843541189447, tol);
    EXPECT_NEAR(turbAy, -2.3313952836997660 , tol);
    EXPECT_NEAR(turbAz, -2.5229059800250133, tol);

}

TEST(Turbulence, stirParticle3)
{
    /* test stirParticle
     *
     * 1. define an (x,y,z) particle
     * 2. create a few modes, amplitudes and phases
     * 3. apply stirring and check resulting accelerations
     *
     * choose your modes, phases and amplitudes wisely. use synthetic values that have a well-defined
     * effect on the resulting accelerations that's easy to reason about.
     *
     * Examples:
     * -if there is only one mode, the accelerations are proportional to the amplitudes
     * -the acceleration sum of two separate calls with modes m1 and m2 should be the same
     *  as one call with both modes
     *
     */
    using Tc = double;
    using T = double;
    using Ta = double;

    T tol = 1.e-8;
    size_t ndim = 3;

    Tc xi[1] = {-0.2};
    Tc yi[1] = {0.3};
    Tc zi[1] = {-0.4};

    int numModes = 2;
    T modes[6] ={10.0,15.0,14.0, 12.566370614359172,0,0};
    T phaseReal[6] = {1.1,1.2,1.3, 1.0,1.0,1.0};
    T phaseImag[6] = {0.7,0.8,0.9, 1.0,1.0,1.0};
    T amplitudes[2] = {2.0, 1.0};

    T ax[1] = {0.0};
    T ay[1] = {0.0};
    T az[1] = {0.0};


    sph::computeStirring(0, 1, ndim, xi, yi, zi, ax, ay, az, numModes, modes, phaseReal, phaseImag,
                     amplitudes, 1.0);

    EXPECT_NEAR(ax[0], -2.1398843541189447 -0.22123189208356875, tol);
    EXPECT_NEAR(ay[0], -2.3313952836997660 -0.22123189208356875, tol);
    EXPECT_NEAR(az[0], -2.5229059800250133 -0.22123189208356875, tol);

}


TEST(Turbulence, computePhases)
{

    using Tc = double;
    using T = double;
    using Ta = double;

    T tol = 1.e-8;
    size_t ndim = 3;

    Tc xi[1] = {-0.2};
    Tc yi[1] = {0.3};
    Tc zi[1] = {-0.4};

    size_t numModes = 1;
    std::vector<T> modes ={10.0,15.0,14.0};
    std::vector<T> phasesReal = {0,0,0};
    std::vector<T> phasesImag = {0,0,0};
    std::vector<T> OUPhases = {0.7,0.8,0.9, 1.0,1.0,1.0};
    T solWeight = 0.5;
    T amplitudes[1] = {2.0};

sph::computePhases(numModes, ndim, OUPhases, solWeight, modes, phasesReal, phasesImag);

    EXPECT_NEAR(phasesReal[0], 0.34999999403953552, tol); //actual value 0.35000000000000000
    EXPECT_NEAR(phasesReal[1], 0.44999998807907104, tol); //actual value 0.45000000000000000
    EXPECT_NEAR(phasesReal[2], 0.50000000000000000, tol); //actual value 0.50000000000000000
    EXPECT_NEAR(phasesImag[0], 0.40000000596046448, tol); //actual value 0.40000000000000000
    EXPECT_NEAR(phasesImag[1], 0.50000000000000000, tol); //actual value 0.50000000000000000
    EXPECT_NEAR(phasesImag[2], 0.50000000000000000, tol); //actual value 0.50000000000000000
}

TEST(Turbulence, updateNoise)
{

    using Tc = double;
    using T = double;
    using Ta = double;

    std::vector<T> OUPhases = {0.7,0.8,0.9, 1.0,1.0,1.0};

    //sph::updateNoise(std::vector<T>& phases, T stddev, T dt, T ts, std::mt19937& gen)

    //constants well-defined, ask for integration method
  // do it for a single phase

}
