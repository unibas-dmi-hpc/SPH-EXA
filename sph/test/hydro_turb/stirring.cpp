
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

    EXPECT_TRUE(false);
}

TEST(Turbulence, computePhases)
{
    EXPECT_TRUE(false);
}
