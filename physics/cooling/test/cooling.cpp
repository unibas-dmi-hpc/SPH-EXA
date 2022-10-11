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
#define CONFIG_BFLOAT_8
#include <iostream>
#include <vector>

#include "gtest/gtest.h"

//#include "grackle_deps/version.h"
#include "cooling/include/cooling.hpp"

TEST(cooling_grackle, test1a)
{
    using Real = double;

    constexpr Real density_units = 1.67e-24;
    constexpr Real time_units = 1.0e12;
    constexpr Real length_units = 1.0;

    constexpr Real GCGS = 6.674e-8;

    constexpr Real density_units_c = 1. / (time_units * time_units);
    //EXPECT_NEAR(density_units, density_units_c, 1e-26);
    printf("density units: %g\t%g\n", density_units, density_units_c);
    constexpr double MSOLG = 1.989e33;
    const double KPCCM = 3.086e21;

    const Real mass_unit = std::pow(length_units, 3.0) * density_units / MSOLG;

    cooling::cooling_data<Real> cd(PROJECT_SOURCE_DIR "/physics/cooling/test/unittest.test", mass_unit, 1.0 / KPCCM, 0, time_units);


   constexpr gr_float tiny_number = 1.e-20;
   constexpr Real     dt          = 3.15e7 * 1e6; // grackle_units.time_units;
   constexpr Real     mh          = 1.67262171e-24;
   constexpr Real     kboltz      = 1.3806504e-16;

   auto rho = std::vector<Real>{1.0};
   Real temperature_units =
       mh *
       pow(cd.global_values.units.a_units * cd.global_values.units.length_units / cd.global_values.units.time_units, 2.) /
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

   cooling::cool_particle<Real>(cd.global_values,
                          dt / cd.global_values.units.time_units,
                 rho[0],
                 u[0],
                 HI_fraction[0],
                 HII_fraction[0],
                 HM_fraction[0],
                 HeI_fraction[0],
                 HeII_fraction[0],
                 HeIII_fraction[0],
                 H2I_fraction[0],
                 H2II_fraction[0],
                 DI_fraction[0],
                 DII_fraction[0],
                 HDI_fraction[0],
                 e_fraction[0],
                 metal_fraction[0],
                 volumetric_heating_rate[0],
                 specific_heating_rate[0],
                 RT_heating_rate[0],
                 RT_HI_ionization_rate[0],
                 RT_HeI_ionization_rate[0],
                 RT_HeII_ionization_rate[0],
                 RT_H2_dissociation_rate[0],
                 H2_self_shielding_length[0]);

   std::cout << HI_fraction[0] << std::endl;

   EXPECT_NEAR(HI_fraction[0], 0.630705, 1e-6);
   EXPECT_NEAR(u[0], 2.95159e+35, 1e30);

   //cleanGrackle();
}
TEST(cooling_grackle2, test2)
{
    using Real = double;
    cooling::cooling_data<Real> cd("~/param.test");

    /*Parameters set:
   with_radiative_cooling = 1;
   primordial_chemistry   = 1;
   dust_chemistry         = 0;
   metal_cooling          = 0;
   UVbackground           = 0;*/

   constexpr gr_float tiny_number = 1.e-20;
   constexpr Real     dt          = 0.01; // grackle_units.time_units;


   size_t n_rho = 100;
   size_t n_u = 100;
   Real rho_min_log = -2;
   Real rho_max_log = 3;
   Real u_min_log = -3;
   Real u_max_log = 1.5;

   std::vector<Real> rho_vec(n_rho);
   std::vector<Real> u_vec(n_u);
   for (size_t i = 0; i < n_rho; i++) {
       Real val = (rho_max_log - rho_min_log) / n_rho * i + rho_min_log;
       rho_vec[i] = std::pow(10., val);
   }
   for (size_t i = 0; i < n_u; i++) {
       Real val = (u_max_log - u_min_log) / n_u * i + u_min_log;
       u_vec[i] = std::pow(10., val);
   }


   auto cool_test_data = [&dt, &cd](Real rho_in, Real u_in)
   {
       auto rho                      = std::vector<Real>{rho_in};
       auto u                        = std::vector<Real>{u_in};
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
       auto metal_fraction           = std::vector<Real>{tiny_number};
       auto volumetric_heating_rate  = std::vector<Real>{0.};
       auto specific_heating_rate    = std::vector<Real>{0.};
       auto RT_heating_rate          = std::vector<Real>{0.};
       auto RT_HI_ionization_rate    = std::vector<Real>{0.};
       auto RT_HeI_ionization_rate   = std::vector<Real>{0.};
       auto RT_HeII_ionization_rate  = std::vector<Real>{0.};
       auto RT_H2_dissociation_rate  = std::vector<Real>{0.};
       auto H2_self_shielding_length = std::vector<Real>{0.};


       cooling::cool_particle(cd.global_values,
                              dt,
                     rho[0],
                     u[0],
                     HI_fraction[0],
                     HII_fraction[0],
                     HM_fraction[0],
                     HeI_fraction[0],
                     HeII_fraction[0],
                     HeIII_fraction[0],
                     H2I_fraction[0],
                     H2II_fraction[0],
                     DI_fraction[0],
                     DII_fraction[0],
                     HDI_fraction[0],
                     e_fraction[0],
                     metal_fraction[0],
                     volumetric_heating_rate[0],
                     specific_heating_rate[0],
                     RT_heating_rate[0],
                     RT_HI_ionization_rate[0],
                     RT_HeI_ionization_rate[0],
                     RT_HeII_ionization_rate[0],
                     RT_H2_dissociation_rate[0],
                     H2_self_shielding_length[0]);


       return u[0];
   };
   std::vector<Real> results(n_rho * n_u);
   std::FILE *file = std::fopen("~/cooling_test1/sphexa.txt", "w");
   for (size_t i = 0; i < n_rho; i++) {
       for (size_t k = 0; k < n_u; k++) {
           //size_t it = k + i * n_u;
           Real u_cooled = cool_test_data(rho_vec[i], u_vec[k]);
           std::fprintf(file, "%g %g %g\n", rho_vec[i], u_vec[k], u_cooled);
       }
   }
   std::fclose(file);

}
