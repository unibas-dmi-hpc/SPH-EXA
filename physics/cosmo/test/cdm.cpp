#include "gtest/gtest.h"

#include "cosmo/cosmology_data.hpp"
#include "cosmo/cdm.hpp"

TEST(cosmo_cdm, constructor)
{
    using T = double; 
    using Cosmo = cosmo::CDM<T>;

    EXPECT_NO_THROW(Cosmo({.H0=1, .OmegaMatter=1}));
    EXPECT_NO_THROW(Cosmo({.H0=1, .OmegaMatter=0}));
    EXPECT_THROW(   Cosmo({.H0=0, .OmegaMatter=0}), std::domain_error);
    EXPECT_THROW(   Cosmo({.H0=0, .OmegaMatter=1}), std::domain_error);

    EXPECT_THROW(   Cosmo({.H0=1, .OmegaMatter=-1}), std::domain_error);
}

TEST(cosmo_cdm, hubbleRate)
{
    using T = double; 
    using Cosmo = cosmo::CDM<T>;

    EXPECT_EQ(Cosmo({.H0=1, .OmegaMatter=1}).H(1), 1);
    EXPECT_EQ(Cosmo({.H0=2, .OmegaMatter=1}).H(1), 2);

    EXPECT_NEAR(Cosmo({.H0=4, .OmegaMatter=0.1}).H(0.5), 16*sqrt(0.1*0.5 + (1-0.1)*pow(0.5,2)), 1e-12);
    EXPECT_NEAR(Cosmo({.H0=4, .OmegaMatter=0.3}).H(0.5), 16*sqrt(0.3*0.5 + (1-0.3)*pow(0.5,2)), 1e-12);
}

TEST(cosmo_cdm, time)
{
    using namespace cosmo;
    using T = double; 
    using Cosmo = cosmo::CDM<T>;

    {
    Cosmo c({.H0=1, .OmegaMatter=1});
    }

    {
    Cosmo c({.H0=1, .OmegaMatter=0});
    }

    {
    Cosmo c({.H0=9, .OmegaMatter=0.2});
    }
}

TEST(cosmo_cdm, scale_factor_function)
{
    using namespace cosmo;
    using T = double; 
    using Cosmo = cosmo::CDM<T>;

    {
    Cosmo c({.H0=1, .OmegaMatter=1});
    }

    {
    Cosmo c({.H0=1, .OmegaMatter=0});
    }

    {
    Cosmo c({.H0=9, .OmegaMatter=0.2});
    }
}

TEST(cosmo_cdm, drift_time_correction)
{
    using namespace cosmo;
    using T = double; 
    using Cosmo = cosmo::CDM<T>;

    {
    Cosmo c({.H0=72, .OmegaMatter=1});
    }
}

TEST(cosmo_cdm, kick_time_correction)
{
    using namespace cosmo;
    using T = double; 
    using Cosmo = cosmo::CDM<T>;

    {
    Cosmo c({.H0=72, .OmegaMatter=1});
    }
}
