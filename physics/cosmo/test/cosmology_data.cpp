#include "cosmo/cosmology_data.hpp"
#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>
#include "gtest/gtest.h"
#include "gtest/gtest-spi.h"
#include <csignal>

TEST(cosmo, hubbleRate)
{
    using T = double; 
    using Cosmo = cosmo::CosmologyData<T>;
    (void)(::testing::GTEST_FLAG(death_test_style) = "threadsafe");

    EXPECT_NE(Cosmo({.H0=0}).H(0), Cosmo({.H0=0}).H(0));
    EXPECT_NE(Cosmo({.H0=1}).H(0), Cosmo({.H0=1}).H(0));

    EXPECT_EQ(Cosmo({.H0=0}).H(1), 0);
    EXPECT_EQ(Cosmo({.H0=1, .OmegaMatter=1}).H(1), 1);
    EXPECT_EQ(Cosmo({.H0=2, .OmegaMatter=1}).H(1), 2);

    EXPECT_NEAR(Cosmo({.H0=4, .OmegaMatter=1}).H(0.5), 16*sqrt(0.5), 1e-12);
    EXPECT_NEAR(Cosmo({.H0=4, .OmegaRadiation=1}).H(0.5), 16*sqrt(1), 1e-12);
    EXPECT_NEAR(Cosmo({.H0=4, .OmegaLambda=1}).H(0.5), 16*sqrt(pow(0.5,4)), 1e-12);

    EXPECT_NEAR(Cosmo({.H0=4, .OmegaMatter=0.3}).H(0.5), 16*sqrt(0.3*0.5 + (1-0.3)*pow(0.5,2)), 1e-12);
    EXPECT_NEAR(Cosmo({.H0=4, .OmegaRadiation=0.3}).H(0.5), 16*sqrt(0.3 + (1-0.3)*pow(0.5,2)), 1e-12);
    EXPECT_NEAR(Cosmo({.H0=4, .OmegaLambda=0.3}).H(0.5), 16*sqrt(0.3*pow(0.5,4) + (1-0.3)*pow(0.5,2)), 1e-12);


    EXPECT_NEAR(Cosmo({.H0=4, .OmegaMatter=0.3, .OmegaRadiation=0.2}).H(0.5), 
            16 * sqrt(0.2 + 0.3*0.5 + (1 - 0.3 - 0.2) * pow(0.5,2)), 1e-12);

    EXPECT_NEAR(Cosmo({.H0=4, .OmegaMatter=0.3, .OmegaLambda=0.2}).H(0.5), 
            16 * sqrt(0.2*pow(0.5,4) + 0.3*0.5 + (1 - 0.3 - 0.2) * pow(0.5,2)), 1e-12);

    EXPECT_NEAR(Cosmo({.H0=4, .OmegaRadiation=0.3, .OmegaLambda=0.2}).H(0.5), 
            16 * sqrt(0.2*pow(0.5,4) + 0.3 + (1 - 0.3 - 0.2) * pow(0.5,2)), 1e-12);

    EXPECT_NEAR(Cosmo({.H0=4, .OmegaMatter=0.1, .OmegaRadiation=0.3, .OmegaLambda=0.2}).H(0.5), 
            16 * sqrt(0.1*0.5 + 0.2*pow(0.5,4) + 0.3 + (1 - 0.3 - 0.2 - 0.1) * pow(0.5,2)), 1e-12);
}

TEST(cosmo, romberg)
{
    using namespace cosmo;
    using T = double; 

    EXPECT_NEAR(romberg<T>([](T x){ return 1.0; }, 0.0, 10.0, 1e-7), 10.0, 1e-7);
    EXPECT_NEAR(romberg<T>([](T x){ return 1.0; }, -10, 10.0, 1e-7), 20.0, 1e-7);
    EXPECT_NEAR(romberg<T>([](T x){ return   x; }, -10, 10.0, 1e-8),  0.0, 1e-7);
    EXPECT_NEAR(romberg<T>([](T x){ return   x*x; }, -1, 1.0, 1e-7), 2.0/3.0, 1e-7);
    EXPECT_NEAR(romberg<T>([](T x){ return 1/x; }, 1, 2, 1e-7),  log(2), 1e-7);
    EXPECT_NEAR(romberg<T>([](T x){ return 1/(x*x); }, 0.01, 1, 1e-7),  99.0, 1e-7);
    EXPECT_NEAR(romberg<T>([](T x){ return 1/(x*x); }, 0.01, 9, 1e-7),  99.88888888, 1e-7);

    EXPECT_NONFATAL_FAILURE(EXPECT_NEAR(romberg<T>([](T x){ return 1/x; }, 1, 2, 1e-4),  log(2), 1e-12), "");
    EXPECT_NONFATAL_FAILURE(EXPECT_NEAR(romberg<T>([](T x){ return 1/x; }, 1, 2, 1e-5),  log(2), 1e-12), "");
    EXPECT_NONFATAL_FAILURE(EXPECT_NEAR(romberg<T>([](T x){ return 1/x; }, 1, 2, 1e-8),  log(2), 1e-15), "");

}

TEST(cosmo, time)
{
    using namespace cosmo;
    using T = double; 
    using Cosmo = cosmo::CosmologyData<T>;

    {
    Cosmo c({.H0=1, .OmegaMatter=1, .OmegaRadiation=0, .OmegaLambda=0});
    EXPECT_NEAR(c.time(1.0), 0.6666666666666666, 1e-7);
    EXPECT_NEAR(c.time(0.5), 0.2357022603955158, 1e-7);
    }

    {
    Cosmo c({.H0=9, .OmegaMatter=0.2, .OmegaRadiation=0.1, .OmegaLambda=0.5});
    EXPECT_NEAR(c.time(1.0), 0.08807284676134755, 1e-7);
    EXPECT_NEAR(c.time(0.5), 0.03158438848551350, 1e-7);
    EXPECT_NEAR(c.time(0.001), 1.7556501496559042e-07, 1e-7);
    }
}

TEST(cosmo, newton)
{
    using namespace cosmo;
    using T = double; 

    {
    auto f  = [](const T x){ printf("f : %23.15e\n", x*x - 2.0); return x*x - 2.0; };
    auto fp = [](const T x){ printf("f': %23.15e\n", 2.0*x); return 2.0*x; };

    EXPECT_NEAR(newton<T>(f, fp,  1.0, -10.0, 10.0, 1e-7), sqrt(2), 1e-7);
    EXPECT_NEAR(newton<T>(f, fp, -2.0, -10.0, 10.0, 1e-7), -sqrt(2), 1e-7);
    //EXPECT_NEAR(newton<T>(f, fp, 0, 0, 10, 1e-7), sqrt(2), 1e-7);
    }

    {
    auto f  = [](const T x){ return x - 2 + log(x); };
    auto fp = [](const T x){ return 1 + 1/x; };
    EXPECT_NEAR(newton<T>(f, fp,  1.5, -10, 10, 1e-7), 1.5571455989976113, 1e-7);
    }

    {
    auto f  = [](const T x){ return sin(x) - (x+1)/(x-1); };
    auto fp = [](const T x){ return cos(x) + 2/((x-1)*(x-1)); };
    EXPECT_NEAR(newton<T>(f, fp,  -0.2, -10, 10, 1e-7), -0.42036240721563467, 1e-7);
    }
}


TEST(cosmo, scale_factor_function)
{
    using namespace cosmo;
    using T = double; 
    using Cosmo = cosmo::CosmologyData<T>;

    {
    //Cosmo c;
    Cosmo c({.H0=1, .OmegaMatter=1, .OmegaRadiation=0, .OmegaLambda=0});
    EXPECT_NEAR(c.a(0.0001), 1, 1e-7);
    EXPECT_NEAR(c.a(1.0), 1, 1e-7);
    }
}

//TEST(cosmo, time)
//{
//    using namespace cosmo;
//    using T = double; 
//
//    using Cosmo = cosmo::CosmologyData<T>;
//    //using CosmoData = cosmo::CData<T>;
//    //
//    (void)(::testing::GTEST_FLAG(death_test_style) = "threadsafe");
//
//    Cosmo c;
//
//    EXPECT_NEAR(c.time(1.0), c.H0, 1e-12);
//
//}
