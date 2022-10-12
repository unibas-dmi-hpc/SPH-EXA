/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief Radiative cooling tests with GRACKLE
 *
 */

#include <iostream>
#include <vector>

#include "gtest/gtest.h"

#include "grackle_deps/version.h"
#include "cooling/cooling.hpp"

TEST(cooling_grackle, test1a)
{
    grackle_options options;
    options.with_radiative_cooling = 1;
    options.primordial_chemistry   = 3;
    options.dust_chemistry         = 1;
    options.metal_cooling          = 1;
    options.UVbackground           = 1;

    initGrackle(PROJECT_SOURCE_DIR "/grackle_repo/input/CloudyData_UVB=HM2012.h5", options);

    using Real = double;

    constexpr gr_float tiny_number = 1.e-20;
    constexpr Real     dt          = 3.15e7 * 1e6; // grackle_units.time_units;
    constexpr Real     mh          = 1.67262171e-24;
    constexpr Real     kboltz      = 1.3806504e-16;

    auto rho = std::vector<Real>{1.0};
    Real temperature_units =
        mh *
        pow(code_units_simulation.a_units * code_units_simulation.length_units / code_units_simulation.time_units, 2.) /
        kboltz;
    auto u                        = std::vector<Real>{1000. / temperature_units};
    auto HI_fraction              = std::vector<Real>{0.76};
    auto HII_fraction             = std::vector<Real>{tiny_number};
    auto HM_fraction              = std::vector<Real>{tiny_number};
    auto HeI_fraction             = std::vector<Real>{0.24};
    auto HeII_fraction            = std::vector<Real>{tiny_number};
    auto HeIII_fraction           = std::vector<Real>{tiny_number};
    auto H2I_fraction             = std::vector<Real>{tiny_number};
    auto H2II_fraction            = std::vector<Real>{tiny_number};
    auto DI_fraction              = std::vector<Real>{2.0 * 3.4e-5};
    auto DII_fraction             = std::vector<Real>{tiny_number};
    auto HDI_fraction             = std::vector<Real>{tiny_number};
    auto e_fraction               = std::vector<Real>{tiny_number};
    auto metal_fraction           = std::vector<Real>{0.01295};
    auto volumetric_heating_rate  = std::vector<Real>{0.};
    auto specific_heating_rate    = std::vector<Real>{0.};
    auto RT_heating_rate          = std::vector<Real>{0.};
    auto RT_HI_ionization_rate    = std::vector<Real>{0.};
    auto RT_HeI_ionization_rate   = std::vector<Real>{0.};
    auto RT_HeII_ionization_rate  = std::vector<Real>{0.};
    auto RT_H2_dissociation_rate  = std::vector<Real>{0.};
    auto H2_self_shielding_length = std::vector<Real>{0.};

    std::cout << HI_fraction[0] << std::endl;
    std::cout << HeI_fraction[0] << std::endl;
    std::cout << metal_fraction[0] << std::endl;

    cool_particle(dt, rho[0], u[0], HI_fraction[0], HII_fraction[0], HM_fraction[0], HeI_fraction[0], HeII_fraction[0],
                  HeIII_fraction[0], H2I_fraction[0], H2II_fraction[0], DI_fraction[0], DII_fraction[0],
                  HDI_fraction[0], e_fraction[0], metal_fraction[0], volumetric_heating_rate[0],
                  specific_heating_rate[0], RT_heating_rate[0], RT_HI_ionization_rate[0], RT_HeI_ionization_rate[0],
                  RT_HeII_ionization_rate[0], RT_H2_dissociation_rate[0], H2_self_shielding_length[0]);

    std::cout << HI_fraction[0] << std::endl;

    EXPECT_NEAR(HI_fraction[0], 0.630705, 1e-6);
    EXPECT_NEAR(u[0], 2.95159e+35, 1e30);

    cleanGrackle();
}
