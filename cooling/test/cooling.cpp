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
#include "cooling.hpp"


TEST(cooling_grackle, test1a)
{

    grackle_options options;
    options.with_radiative_cooling = 1;
    options.primordial_chemistry   = 3;
    options.dust_chemistry         = 1;
    options.metal_cooling          = 1;
    options.UVbackground           = 1;

    // WTF ?!
    //initGrackle("/Users/noah/Documents/Changa/grackle/grackle/grackle_data_files/input/CloudyData_UVB=HM2012.h5",
    //            options);

    using Real = double;

    double   dt          = 3.15e7 * 1e6; // grackle_units.time_units;

    constexpr gr_float tiny_number = 1.e-20;
    constexpr Real mh 1.67262171e-24
    constexpr Real kboltz 1.3806504e-16

    d.dt  = std::vector<Real>{dt};
    d.x   = std::vector<Real>{0.};
    d.y   = std::vector<Real>{0.};
    d.z   = std::vector<Real>{0.};
    d.rho = std::vector<Real>{1.0};
    double temperature_units =
        mh * pow(code_units_simulation.a_units * code_units_simulation.length_units / code_units_simulation.time_units, 2.) / kboltz;
    d.u              = std::vector<Real>{1000. / temperature_units};
    d.HI_fraction    = std::vector<Real>{0.76};
    d.HII_fraction   = std::vector<Real>{tiny_number};
    d.HM_fraction    = std::vector<Real>{tiny_number};
    d.HeI_fraction   = std::vector<Real>{0.24};
    d.HeII_fraction  = std::vector<Real>{tiny_number};
    d.HeIII_fraction = std::vector<Real>{tiny_number};
    d.H2I_fraction   = std::vector<Real>{tiny_number};
    d.H2II_fraction  = std::vector<Real>{tiny_number};
    d.DI_fraction    = std::vector<Real>{2.0 * 3.4e-5};
    d.DII_fraction   = std::vector<Real>{tiny_number};
    d.HDI_fraction   = std::vector<Real>{tiny_number};
    d.e_fraction     = std::vector<Real>{tiny_number};
    d.metal_fraction = std::vector<Real>{0.01295};
    d.volumetric_heating_rate = std::vector<Real>{0.};
    d.specific_heating_rate = std::vector<Real>{0.};
    d.RT_heating_rate = std::vector<Real>{0.};
    d.RT_HI_ionization_rate = std::vector<Real>{0.};
    d.RT_HeI_ionization_rate = std::vector<Real>{0.};
    d.RT_HeII_ionization_rate = std::vector<Real>{0.};
    d.RT_H2_dissociation_rate = std::vector<Real>{0.};
    d.H2_self_shielding_length = std::vector<Real>{0.};
    d.n              = 1;

    std::cout << d.HI_fraction[0] << std::endl;
    std::cout << d.HeI_fraction[0] << std::endl;
    std::cout << d.metal_fraction[0] << std::endl;

    cool_particle<Real, Dataset>(d.dt[0],
                                 d.rho[0],
                                 d.u[0],
                                 d.HI_fraction[0],
                                 d.HII_fraction[0],
                                 d.HM_fraction[0],
                                 d.HeI_fraction[0],
                                 d.HeII_fraction[0],
                                 d.HeIII_fraction[0],
                                 d.H2I_fraction[0],
                                 d.H2II_fraction[0],
                                 d.DI_fraction[0],
                                 d.DII_fraction[0],
                                 d.HDI_fraction[0],
                                 d.e_fraction[0],
                                 d.metal_fraction[0],
                                 d.volumetric_heating_rate[0],
                                 d.specific_heating_rate[0],
                                 d.RT_heating_rate[0],
                                 d.RT_HI_ionization_rate[0],
                                 d.RT_HeI_ionization_rate[0],
                                 d.RT_HeII_ionization_rate[0],
                                 d.RT_H2_dissociation_rate[0],
                                 d.H2_self_shielding_length[0]
    );

    std::cout << d.HI_fraction[0] << std::endl;

    EXPECT_NEAR(d.HI_fraction[0], 0.630705, 1e-6);
    EXPECT_NEAR(d.u[0], 2.95159e+35, 1e30);

    cleanGrackle();
}