#include "gtest/gtest.h"

#include "cosmo/cosmology_data.hpp"
#include "cosmo/static.hpp"

TEST(cosmo_static, drift_time_correction)
{
    using namespace cosmo;
    using T = double; 
    using Cosmo = cosmo::StaticUniverse<T>;

    Cosmo c;

    EXPECT_EQ(c.driftTimeCorrection(1, 0.5), 0.5);
    EXPECT_EQ(c.driftTimeCorrection(1, 0.125), 0.125);

    EXPECT_EQ(c.driftTimeCorrection(0, 0.5), 0.5);
    EXPECT_EQ(c.driftTimeCorrection(0, 0.125), 0.125);
}

TEST(cosmo_static, kick_time_correction)
{
    using namespace cosmo;
    using T = double; 
    using Cosmo = cosmo::StaticUniverse<T>;

    Cosmo c;

    EXPECT_EQ(c.kickTimeCorrection(1, 0.5), 0.5);
    EXPECT_EQ(c.kickTimeCorrection(1, 0.125), 0.125);

    EXPECT_EQ(c.kickTimeCorrection(0, 0.5), 0.5);
    EXPECT_EQ(c.kickTimeCorrection(0, 0.125), 0.125);
}

