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

    EXPECT_NO_THROW(Cosmo({.H0=1, .OmegaRadiation=1}));
    EXPECT_THROW(   Cosmo({.H0=1, .OmegaLambda=1}), std::domain_error);

    EXPECT_NO_THROW(Cosmo({.H0=1, .OmegaMatter=0.5, .OmegaLambda=0.5}));
    //EXPECT_NO_THROW(Cosmo({.H0=1, .OmegaMatter=2}));
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

