#include "gtest/gtest.h"

#include "cosmo/cosmology_data.hpp"
#include "cosmo/lcdm.hpp"

TEST(cosmo_lcdm, constructor)
{
    using T = double; 
    using Cosmo = cosmo::LambdaCDM<T>;

    EXPECT_NO_THROW(Cosmo({.H0=1, .OmegaRadiation=1}));
    EXPECT_THROW(   Cosmo({.H0=1, .OmegaLambda=1}), std::domain_error);

    EXPECT_NO_THROW(Cosmo({.H0=1, .OmegaMatter=0.5, .OmegaLambda=0.5}));
    EXPECT_THROW(Cosmo({.H0=1, .OmegaMatter=1}), std::domain_error);

    EXPECT_THROW(Cosmo({.H0=1, .OmegaMatter=-1}), std::domain_error); 
    EXPECT_THROW(Cosmo({.H0=1, .OmegaRadiation=-1}), std::domain_error); 
    EXPECT_THROW(Cosmo({.H0=1, .OmegaMatter=1, .OmegaRadiation=-1}), std::domain_error); 

    EXPECT_THROW(Cosmo({.H0=1, .OmegaMatter=0, .OmegaLambda=1}), std::domain_error); 

    EXPECT_NO_THROW(Cosmo({.H0=1, .OmegaMatter=2, .OmegaLambda=0.5}));

    Cosmo c(Cosmo::Planck2018);
    EXPECT_EQ(c.H0, Cosmo::Planck2018.H0);
    EXPECT_EQ(c.OmegaMatter, Cosmo::Planck2018.OmegaMatter);
    EXPECT_EQ(c.OmegaRadiation, Cosmo::Planck2018.OmegaRadiation);
    EXPECT_EQ(c.OmegaLambda, Cosmo::Planck2018.OmegaLambda);
}

TEST(cosmo_lcdm, hubbleRate)
{
    using T = double; 
    using Cosmo = cosmo::LambdaCDM<T>;

    EXPECT_EQ(Cosmo({.H0=1, .OmegaRadiation=1}).H(1), 1);
    EXPECT_EQ(Cosmo({.H0=2, .OmegaRadiation=1}).H(1), 2);

    EXPECT_NEAR(Cosmo({.H0=4, .OmegaMatter=0.1, .OmegaRadiation=0.9}).H(0.5), 15.594870951694343, 1e-12);
    EXPECT_NEAR(Cosmo({.H0=4, .OmegaMatter=0.1, .OmegaLambda=1}).H(0.5), 4.7328638264796927, 1e-12);

    EXPECT_NEAR(Cosmo({.H0=4, .OmegaRadiation=0.3}).H(0.5), 16*sqrt(0.3 + (1-0.3)*pow(0.5,2)), 1e-12);
    EXPECT_NEAR(Cosmo({.H0=4, .OmegaMatter=0.1, .OmegaRadiation=0.3}).H(0.5), 16*sqrt(0.3 + 0.1*0.5 + (1-0.3-0.1)*pow(0.5,2)), 1e-12);

    EXPECT_NEAR(Cosmo({.H0=4, .OmegaMatter=0.3, .OmegaRadiation=0.2}).H(0.5), 
            16 * sqrt(0.2 + 0.3*0.5 + (1 - 0.3 - 0.2) * pow(0.5,2)), 1e-12);

    EXPECT_NEAR(Cosmo({.H0=4, .OmegaMatter=0.3, .OmegaLambda=0.2}).H(0.5), 
            16 * sqrt(0.2*pow(0.5,4) + 0.3*0.5 + (1 - 0.3 - 0.2) * pow(0.5,2)), 1e-12);

    EXPECT_NEAR(Cosmo({.H0=4, .OmegaRadiation=0.3, .OmegaLambda=0.2}).H(0.5), 
            16 * sqrt(0.2*pow(0.5,4) + 0.3 + (1 - 0.3 - 0.2) * pow(0.5,2)), 1e-12);

    EXPECT_NEAR(Cosmo({.H0=4, .OmegaMatter=0.1, .OmegaRadiation=0.3, .OmegaLambda=0.2}).H(0.5), 
            16 * sqrt(0.1*0.5 + 0.2*pow(0.5,4) + 0.3 + (1 - 0.3 - 0.2 - 0.1) * pow(0.5,2)), 1e-12);
}

TEST(cosmo_lcdm, time)
{
    using namespace cosmo;
    using T = double; 
    using Cosmo = cosmo::LambdaCDM<T>;

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

TEST(cosmo_lcdm, scale_factor_function)
{
    using namespace cosmo;
    using T = double; 
    using Cosmo = cosmo::LambdaCDM<T>;

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

TEST(cosmo_lcdm, drift_time_correction)
{
    using namespace cosmo;
    using T = double; 
    using Cosmo = cosmo::LambdaCDM<T>;

    {
    Cosmo c(Cosmo::Planck2018);
    EXPECT_EQ(c.driftTimeCorrection(c.time(1), 0), 0);
    EXPECT_NEAR(c.driftTimeCorrection(c.time(0.5), 0.01), 0.018107112953122475, 1e-7);
    EXPECT_NEAR(c.driftTimeCorrection(c.time(1), 0.01), 0.0056673220256661288, 1e-7);
    }
}

TEST(cosmo_lcdm, kick_time_correction)
{
    using namespace cosmo;
    using T = double; 
    using Cosmo = cosmo::LambdaCDM<T>;

    {
    Cosmo c(Cosmo::Planck2018);
    EXPECT_EQ(c.kickTimeCorrection(c.time(1), 0), 0);
    EXPECT_NEAR(c.kickTimeCorrection(c.time(0.5), 0.01), 0.01309800803678838, 1e-7);
    EXPECT_NEAR(c.kickTimeCorrection(c.time(1), 0.01), 0.00741291066657522, 1e-7);
    }
}
