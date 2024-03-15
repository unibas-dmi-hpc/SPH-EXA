
#include <vector>
#include <iostream>

#include "gtest/gtest.h"

#include "sph/hydro_turb/create_modes_spf1.hpp"

//! @brief test determination of numModes
TEST(Turbulence, numModes)
{
    EXPECT_TRUE(false);
}

TEST(Turbulence, spectForm1_unifdimensions)
{
    using T = double;
    /* choose input parameters
     * T Lx, T Ly, T Lz, T stirMax, T stirMin, size_t ndim,
                         size_t spectForm, T powerLawExp, T anglesExp

        for a few cases such that you can test the resulting modes and amplitudes
     */
    const T twopi = 6.283185307179586;
    const T tol=1.e-8;

    T Lx = 1.0;
    T Ly = 1.0;
    T Lz = 1.0;
    T stirMax =  3.000000000000000*twopi/Lx;
    T stirMin = (1.0)*twopi/Lx;
    //stirMin = 6.2831854820251365;
    //stirMax = 18.849556446075539;
    //std::cout << stirMax - 18.849556446075539 << ' ' << stirMin - 6.2831854820251365;
    //std::cout << T(2.0*M_PI) ;
    size_t st_maxmodes=100000;
    size_t ndim = 3;
    int numModes = 0;
    T modes[st_maxmodes];
    T amplitudes[st_maxmodes];
    bool verbose = true;

    sph::createStirringModesSpf1(numModes, modes, amplitudes, Lx, Ly, Lz, st_maxmodes, stirMax, stirMin, ndim, verbose);
    
    EXPECT_TRUE(numModes == 112);
                                   
    EXPECT_NEAR(modes[3*(10-1)+0],  0.0000000000000000, tol);
    EXPECT_NEAR(modes[3*(10-1)+1], -0.0000000000000000, tol);
    EXPECT_NEAR(modes[3*(10-1)+2],  18.849556446075439, tol);
    EXPECT_NEAR(0.5*amplitudes[(10-1)],  0.0000000000000000, tol);

    EXPECT_NEAR(modes[3*(54-1)+0],  6.2831854820251465, tol);
    EXPECT_NEAR(modes[3*(54-1)+1], -6.2831854820251465, tol);
    EXPECT_NEAR(modes[3*(54-1)+2],  0.0000000000000000, tol);
    EXPECT_NEAR(0.5*amplitudes[(54-1)], 1.1461712345826693, tol);

    EXPECT_NEAR(modes[3*(78-1)+0],  12.566370964050293, tol);
    EXPECT_NEAR(modes[3*(78-1)+1], -0.0000000000000000, tol);
    EXPECT_NEAR(modes[3*(78-1)+2],  0.0000000000000000, tol);
    EXPECT_NEAR(0.5*amplitudes[(78-1)], 1.0000000000000000, tol);
}

TEST(Turbulence, spectForm1_nonunifdimensions)
{
    using T = double;
    /* choose input parameters
     * T Lx, T Ly, T Lz, T stirMax, T stirMin, size_t ndim,
                         size_t spectForm, T powerLawExp, T anglesExp

        for a few cases such that you can test the resulting modes and amplitudes
     */
    const T twopi = 2.0 * M_PI;
    const T tol=1.e-8;
    T Lx = 0.7;
    T Ly = 1.2;
    T Lz = 1.5;
    T stirMax =  3.000000000000001*twopi/Lx;
    T stirMin = (0.999999999999999)*twopi/Lx;
    //stirMin = 6.2831854820251465;
    //stirMax = 18.849556446075439;
    size_t st_maxmodes=100000;
    size_t ndim = 3;
    int numModes = 0;
    T modes[st_maxmodes];
    T amplitudes[st_maxmodes];
    bool verbose = true;

    sph::createStirringModesSpf1(numModes, modes, amplitudes, Lx, Ly, Lz, st_maxmodes, stirMax, stirMin, ndim, verbose);
    
    EXPECT_TRUE(numModes == 300);
                                   
    EXPECT_NEAR(modes[3*(10-1)+0],  0.0000000000000000, tol);
    EXPECT_NEAR(modes[3*(10-1)+1], -0.0000000000000000, tol);
    EXPECT_NEAR(modes[3*(10-1)+2],  20.943951606750488, tol);
    EXPECT_NEAR(0.5*amplitudes[(10-1)],  0.80812206144596110, tol);

    EXPECT_NEAR(modes[3*(54-1)+0],  0.0000000000000000, tol);
    EXPECT_NEAR(modes[3*(54-1)+1], -10.471975387256329, tol);
    EXPECT_NEAR(modes[3*(54-1)+2],  16.755161285400391, tol);
    EXPECT_NEAR(0.5*amplitudes[(54-1)],  0.88997794370873951, tol);

    EXPECT_NEAR(modes[3*(78-1)+0],   0.0000000000000000, tol);
    EXPECT_NEAR(modes[3*(78-1)+1], -15.707963080884493, tol);
    EXPECT_NEAR(modes[3*(78-1)+2],  16.755161285400391, tol);
    EXPECT_NEAR(0.5*amplitudes[(78-1)], 0.64827464011102576, tol);
                
}

TEST(Turbulence, spectForm2)
{
    /* choose input parameters
     * T Lx, T Ly, T Lz, T stirMax, T stirMin, size_t ndim,
                         size_t spectForm, T powerLawExp, T anglesExp

        for a few cases such that you can test the resulting modes and amplitudes
     */
    using T = double;

    const T twopi = 2.0 * M_PI;

    T Lx = 1.0;
    T Ly = 1.0;
    T Lz = 1.0;
    T stirMax =  3.00000000000001*twopi/Lx;
    T stirMin = (0.99999999999999)*twopi/Lx;
    size_t st_maxmodes=100000;
    size_t ndim = 3;
    int numModes = 0;
    T modes[st_maxmodes];
    T amplitudes[st_maxmodes];
    T powerLawExp = 5./3.;
    T anglesExp = 2.0;
    bool verbose = true;

    EXPECT_TRUE(false);
}
