
#include <iostream>
#include <vector>
#include <cmath>
#include "gtest/gtest.h"

#include "cooling/cooler.hpp"

#include "cstone/fields/field_get.hpp"
#include "io/ifile_io.hpp"

TEST(cooling_grackle, test1a)
{
    using Real = double;

    constexpr Real density_units = 1.67e-24;
    constexpr Real time_units    = 1.0e12;
    constexpr Real length_units  = 1.0;

    constexpr Real GCGS = 6.674e-8;

    constexpr Real density_units_c = 1. / (time_units * time_units);
    // EXPECT_NEAR(density_units, density_units_c, 1e-26);
    printf("density units: %g\t%g\n", density_units, density_units_c);
    constexpr double MSOLG = 1.989e33;
    const double     KPCCM = 3.086e21;

    const Real mass_unit = std::pow(length_units, 3.0) * density_units / MSOLG;

    cooling::Cooler<Real> cd;

    std::map<std::string, double> grackleOptions;
    grackleOptions["cooling::m_code_in_ms"]           = mass_unit;
    grackleOptions["cooling::l_code_in_kpc"]          = 1. / KPCCM;
    grackleOptions["cooling::use_grackle"]            = 1;
    grackleOptions["cooling::with_radiative_cooling"] = 1;
    grackleOptions["cooling::primordial_chemistry"]   = 3;
    grackleOptions["cooling::dust_chemistry"]         = 1;
    grackleOptions["cooling::UVbackground"]           = 1;
    grackleOptions["cooling::metal_cooling"]          = 1;

    sphexa::BuiltinWriter attributeSetter(grackleOptions);
    cd.loadOrStoreAttributes(&attributeSetter);

    cd.init(0, time_units);

    constexpr Real tiny_number = 1.e-20;
    constexpr Real dt          = 3.15e7 * 1e6; // grackle_units.time_units;
    constexpr Real mh          = 1.67262171e-24;
    constexpr Real kboltz      = 1.3806504e-16;

    auto rho = std::vector<Real>{1.0};
    /*Real temperature_units =
            mh *
            pow(cd.get_global_values().units.a_units * cd.get_global_values().units.length_units /
       cd.get_global_values().units.time_units, 2.) / kboltz;*/

    Real temperature_units = mh * std::pow(length_units / time_units, 2.) / kboltz;

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

    auto grData =
        std::tie(HI_fraction, HII_fraction, HM_fraction, HeI_fraction, HeII_fraction, HeIII_fraction, H2I_fraction,
                 H2II_fraction, DI_fraction, DII_fraction, HDI_fraction, e_fraction, metal_fraction,
                 volumetric_heating_rate, specific_heating_rate, RT_heating_rate, RT_HI_ionization_rate,
                 RT_HeI_ionization_rate, RT_HeII_ionization_rate, RT_H2_dissociation_rate, H2_self_shielding_length);

    cd.cool_particle(dt / time_units, rho[0], u[0], cstone::getPointers(grData, 0));

    std::cout << HI_fraction[0] << std::endl;

    EXPECT_NEAR(HI_fraction[0], 0.630705, 1e-6);
    EXPECT_NEAR(u[0], 2.95159e+35, 1e30);
}
// This test just produces a table of cooling values for different choices of rho and u
TEST(cooling_grackle2, test2)
{
    // Path where to write the table
    const std::string writePath{"sphexa_cooling_test.txt"};

    using Real = double;
    cooling::Cooler<Real>         cd;
    std::map<std::string, double> grackleOptions;
    grackleOptions["cooling::m_code_in_ms"]           = 1e16;
    grackleOptions["cooling::l_code_in_kpc"]          = 46400;
    grackleOptions["cooling::use_grackle"]            = 1;
    grackleOptions["cooling::with_radiative_cooling"] = 1;
    grackleOptions["cooling::primordial_chemistry"]   = 1;
    grackleOptions["cooling::dust_chemistry"]         = 0;
    grackleOptions["cooling::UVbackground"]           = 0;
    grackleOptions["cooling::metal_cooling"]          = 0;

    sphexa::BuiltinWriter attributeSetter(grackleOptions);
    cd.loadOrStoreAttributes(&attributeSetter);
    cd.init(0);

    constexpr Real tiny_number = 1.e-20;
    constexpr Real dt          = 0.01; // grackle_units.time_units;

    size_t n_rho       = 100;
    size_t n_u         = 100;
    Real   rho_min_log = -2;
    Real   rho_max_log = 3;
    Real   u_min_log   = -3;
    Real   u_max_log   = 1.5;

    std::vector<Real> rho_vec(n_rho);
    std::vector<Real> u_vec(n_u);
    for (size_t i = 0; i < n_rho; i++)
    {
        Real val   = (rho_max_log - rho_min_log) / n_rho * i + rho_min_log;
        rho_vec[i] = std::pow(10., val);
    }
    for (size_t i = 0; i < n_u; i++)
    {
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

        auto grData = std::tie(HI_fraction, HII_fraction, HM_fraction, HeI_fraction, HeII_fraction, HeIII_fraction,
                               H2I_fraction, H2II_fraction, DI_fraction, DII_fraction, HDI_fraction, e_fraction,
                               metal_fraction, volumetric_heating_rate, specific_heating_rate, RT_heating_rate,
                               RT_HI_ionization_rate, RT_HeI_ionization_rate, RT_HeII_ionization_rate,
                               RT_H2_dissociation_rate, H2_self_shielding_length);

        cd.cool_particle(dt, rho[0], u[0], cstone::getPointers(grData, 0));

        return u[0];
    };
    std::vector<Real> results(n_rho * n_u);
    std::FILE*        file = std::fopen(writePath.c_str(), "w");
    if (!file) throw std::runtime_error("File could not be opened");
    for (size_t i = 0; i < n_rho; i++)
    {
        for (size_t k = 0; k < n_u; k++)
        {
            // size_t it = k + i * n_u;
            Real u_cooled = cool_test_data(rho_vec[i], u_vec[k]);
            std::fprintf(file, "%g %g %g\n", rho_vec[i], u_vec[k], u_cooled);
        }
    }
    std::fclose(file);
}
