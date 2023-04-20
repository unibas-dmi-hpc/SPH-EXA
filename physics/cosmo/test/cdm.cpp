#include "cosmo/cosmology_data.hpp"
#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>
#include "gtest/gtest.h"
#include "gtest/gtest-spi.h"
#include <csignal>

TEST(cosmo, constructor)
{
    using T = double; 
    using Cosmo = cosmo::CDM<T>;

    //EXPECT_NO_THROW(Cosmo(Cosmo::Static));
    //EXPECT_FALSE(Cosmo(Cosmo::Static).isComoving);

    EXPECT_NO_THROW(Cosmo({.H0=1, .OmegaMatter=1}));
    EXPECT_NO_THROW(Cosmo({.H0=1, .OmegaMatter=0}));
    EXPECT_THROW(   Cosmo({.H0=0, .OmegaMatter=0}), std::domain_error);
    EXPECT_THROW(   Cosmo({.H0=0, .OmegaMatter=1}), std::domain_error);

    EXPECT_THROW(   Cosmo({.H0=1, .OmegaMatter=-1}), std::domain_error);
}

TEST(cosmo, hubbleRate)
{
    using T = double; 
    using Cosmo = cosmo::CDM<T>;

    EXPECT_EQ(Cosmo({.H0=1, .OmegaMatter=1}).H(1), 1);
    EXPECT_EQ(Cosmo({.H0=2, .OmegaMatter=1}).H(1), 2);

    EXPECT_NEAR(Cosmo({.H0=4, .OmegaMatter=0.1}).H(0.5), 16*sqrt(0.1*0.5 + (1-0.1)*pow(0.5,2)), 1e-12);
    EXPECT_NEAR(Cosmo({.H0=4, .OmegaMatter=0.3}).H(0.5), 16*sqrt(0.3*0.5 + (1-0.3)*pow(0.5,2)), 1e-12);
}

TEST(cosmo, romberg)
{
    using namespace cosmo;
    using T = double; 

    EXPECT_NEAR(romberg<T>([]([[maybe_unused]] T x){ return 1.0; }, 0.0, 10.0, 1e-7), 10.0, 1e-7);
    EXPECT_NEAR(romberg<T>([]([[maybe_unused]] T x){ return 1.0; }, -10, 10.0, 1e-7), 20.0, 1e-7);
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
    using Cosmo = cosmo::CDM<T>;

    {
    Cosmo c({.H0=1, .OmegaMatter=1, .OmegaRadiation=0, .OmegaLambda=1});
    EXPECT_NEAR(c.time(1.0), 0.7846190892554956, 1e-7);
    EXPECT_NEAR(c.time(0.5), 0.2753584353773286, 1e-7);
    }

    {
    Cosmo c({.H0=1, .OmegaMatter=0, .OmegaRadiation=1, .OmegaLambda=0});
    EXPECT_NEAR(c.time(1.0), 0.5, 1e-7);
    EXPECT_NEAR(c.time(0.5), 1./8., 1e-7);
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
    auto f  = [](const T x){ return x*x - 2.0; };
    auto fp = [](const T x){ return 2.0*x; };
    EXPECT_THROW(newton<T>(f, fp,  1.0, 10.0, -10.0, 1e-7), std::invalid_argument);
    }

    {
    auto f  = [](const T x){ return x*x; };
    auto fp = [](const T x){ return 2.0*x; };

    EXPECT_EQ(newton<T>(f, fp,  0.0, -10.0, 10.0), 0);
    }

    {
    auto f  = [](const T x){ return x*x - 2.0; };
    auto fp = [](const T x){ return 2.0*x; };

    EXPECT_NEAR(newton<T>(f, fp,  1.0, -10.0, 10.0, 1e-7), sqrt(2), 1e-7);
    EXPECT_NEAR(newton<T>(f, fp,  1.0, -10.0, 10.0, 1e-7), sqrt(2), 1e-7);
    EXPECT_NEAR(newton<T>(f, fp, -2.0, -10.0, 10.0, 1e-7), -sqrt(2), 1e-7);
    EXPECT_NEAR(newton<T>(f, fp, 0, 0, 10, 1e-7), sqrt(2), 1e-7);
    EXPECT_NEAR(newton<T>(f, fp, 0, -10, 10, 1e-7), sqrt(2), 1e-7);
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
    using Cosmo = cosmo::CDM<T>;

    {
    Cosmo c({.H0=1, .OmegaMatter=1, .OmegaRadiation=0, .OmegaLambda=1});
    EXPECT_NEAR(c.a(c.time(.001)), .001, 1e-7);
    EXPECT_NEAR(c.a(c.time(0.5)), 0.5, 1e-7);
    EXPECT_NEAR(c.a(c.time(1)), 1, 1e-7);
    }

    {
    Cosmo c({.H0=1, .OmegaMatter=0, .OmegaRadiation=1, .OmegaLambda=0});
    EXPECT_NEAR(c.a(c.time(.001)), .001, 1e-7);
    EXPECT_NEAR(c.a(c.time(0.5)), 0.5, 1e-7);
    EXPECT_NEAR(c.a(c.time(1)), 1, 1e-7);
    }

    {
    Cosmo c({.H0=9, .OmegaMatter=0.2, .OmegaRadiation=0.1, .OmegaLambda=0.5});
    EXPECT_NEAR(c.a(c.time(.001)), .001, 1e-7);
    EXPECT_NEAR(c.a(c.time(0.5)), 0.5, 1e-7);
    EXPECT_NEAR(c.a(c.time(1)), 1, 1e-7);
    }
}

TEST(cosmo, drift_time_correction)
{
    using namespace cosmo;
    using T = double; 
    using Cosmo = cosmo::CDM<T>;

    {
    Cosmo c({.H0=72, .OmegaMatter=1);
    EXPECT_EQ(c.driftTimeCorrection(c.time(1), 0), 0);
    EXPECT_NEAR(c.driftTimeCorrection(c.time(0.5), 0.01), 0.018107112953122475, 1e-7);
    EXPECT_NEAR(c.driftTimeCorrection(c.time(1), 0.01), 0.0056673220256661288, 1e-7);
    }
}

TEST(cosmo, kick_time_correction)
{
    using namespace cosmo;
    using T = double; 
    using Cosmo = cosmo::CDM<T>;

    {
    Cosmo c({.H0=72, .OmegaMatter=1);
    EXPECT_EQ(c.kickTimeCorrection(c.time(1), 0), 0);
    EXPECT_NEAR(c.kickTimeCorrection(c.time(0.5), 0.01), 0.01309800803678838, 1e-7);
    EXPECT_NEAR(c.kickTimeCorrection(c.time(1), 0.01), 0.00741291066657522, 1e-7);
    }
}

TEST(cosmo, static_universe)
{
    using namespace cosmo;
    using T = double; 
    using Cosmo = cosmo::StaticUniverse<T>;

    Cosmo c;

    EXPECT_EQ(c.driftTimeCorrection(1, 0.5), 0.5);
    EXPECT_EQ(c.driftTimeCorrection(1, 0.125), 0.125);

    EXPECT_EQ(c.kickTimeCorrection(1, 0.5), 0.5);
    EXPECT_EQ(c.kickTimeCorrection(1, 0.125), 0.125);
}
