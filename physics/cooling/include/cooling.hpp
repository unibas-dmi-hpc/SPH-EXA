/*
 * MIT License
 *
 * Copyright (c) 2022 CSCS, ETH Zurich
 *               2022 University of Basel
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
 * @brief Interface to the GRACKLE library for radiative cooling
 *
 */

#pragma once

#include <cmath>
#include <optional>
#include <string>

#define CONFIG_BFLOAT_8

extern "C"
{
#include "grackle.h"
}

static constexpr double m_sun = 1.989e33;
static constexpr double pc    = 3.086e18;
static constexpr double G_cgs = 6.674e-8;

static code_units      code_units_simulation;
static chemistry_data* chemistry_data_simulation;

template<typename T>
void cool_particle(const T& dt, T& rho, T& u, T& HI_fraction, T& HII_fraction, T& HM_fraction, T& HeI_fraction,
                   T& HeII_fraction, T& HeIII_fraction, T& H2I_fraction, T& H2II_fraction, T& DI_fraction,
                   T& DII_fraction, T& HDI_fraction, T& e_fraction, T& metal_fraction, T& volumetric_heating_rate,
                   T& specific_heating_rate, T& RT_heating_rate, T& RT_HI_ionization_rate, T& RT_HeI_ionization_rate,
                   T& RT_HeII_ionization_rate, T& RT_H2_dissociation_rate, T& H2_self_shielding_length)
{
    grackle_field_data grackle_fields;
    grackle_fields.grid_rank      = 3;
    int zero[]                    = {0, 0, 0};
    int one[]                     = {1, 1, 1};
    grackle_fields.grid_dimension = one;
    grackle_fields.grid_start     = zero;
    grackle_fields.grid_end       = zero;
    grackle_fields.grid_dx        = 0.0;

    gr_float gr_rho                      = (gr_float)rho;
    grackle_fields.density               = &gr_rho;
    gr_float gr_u                        = (gr_float)u;
    grackle_fields.internal_energy       = &gr_u;
    gr_float x_velocity                  = 0.;
    grackle_fields.x_velocity            = &x_velocity;
    gr_float y_velocity                  = 0.;
    grackle_fields.y_velocity            = &y_velocity;
    gr_float z_velocity                  = 0.;
    grackle_fields.z_velocity            = &z_velocity;
    gr_float HI_density                  = (gr_float)HI_fraction * (gr_float)rho;
    gr_float HII_density                 = (gr_float)HII_fraction * (gr_float)rho;
    gr_float HM_density                  = (gr_float)HM_fraction * (gr_float)rho;
    gr_float HeI_density                 = (gr_float)HeI_fraction * (gr_float)rho;
    gr_float HeII_density                = (gr_float)HeII_fraction * (gr_float)rho;
    gr_float HeIII_density               = (gr_float)HeIII_fraction * (gr_float)rho;
    gr_float H2I_density                 = (gr_float)H2I_fraction * (gr_float)rho;
    gr_float H2II_density                = (gr_float)H2II_fraction * (gr_float)rho;
    gr_float DI_density                  = (gr_float)DI_fraction * (gr_float)rho;
    gr_float DII_density                 = (gr_float)DII_fraction * (gr_float)rho;
    gr_float HDI_density                 = (gr_float)HDI_fraction * (gr_float)rho;
    gr_float e_density                   = (gr_float)e_fraction * (gr_float)rho;
    gr_float metal_density               = (gr_float)metal_fraction * (gr_float)rho;
    gr_float volumetric_heating_rate_gr  = (gr_float)volumetric_heating_rate;
    gr_float specific_heating_rate_gr    = (gr_float)specific_heating_rate;
    gr_float RT_heating_rate_gr          = (gr_float)RT_heating_rate;
    gr_float RT_HI_ionization_rate_gr    = (gr_float)RT_HI_ionization_rate;
    gr_float RT_HeI_ionization_rate_gr   = (gr_float)RT_HeI_ionization_rate;
    gr_float RT_HeII_ionization_rate_gr  = (gr_float)RT_HeII_ionization_rate;
    gr_float RT_H2_dissociation_rate_gr  = (gr_float)RT_H2_dissociation_rate;
    gr_float H2_self_shielding_length_gr = (gr_float)H2_self_shielding_length;

    grackle_fields.HI_density    = &HI_density;
    grackle_fields.HII_density   = &HII_density;
    grackle_fields.HeI_density   = &HeI_density;
    grackle_fields.HeII_density  = &HeII_density;
    grackle_fields.HeIII_density = &HeIII_density;
    grackle_fields.e_density     = &e_density;
    grackle_fields.HM_density    = &HM_density;
    grackle_fields.H2I_density   = &H2I_density;
    grackle_fields.H2II_density  = &H2II_density;
    grackle_fields.DI_density    = &DI_density;
    grackle_fields.DII_density   = &DII_density;
    grackle_fields.HDI_density   = &HDI_density;
    grackle_fields.metal_density = &metal_density;

    grackle_fields.volumetric_heating_rate  = &volumetric_heating_rate_gr;
    grackle_fields.specific_heating_rate    = &specific_heating_rate_gr;
    grackle_fields.RT_heating_rate          = &RT_heating_rate_gr;
    grackle_fields.RT_HI_ionization_rate    = &RT_HI_ionization_rate_gr;
    grackle_fields.RT_HeI_ionization_rate   = &RT_HeI_ionization_rate_gr;
    grackle_fields.RT_HeII_ionization_rate  = &RT_HeII_ionization_rate_gr;
    grackle_fields.RT_H2_dissociation_rate  = &RT_H2_dissociation_rate_gr;
    grackle_fields.H2_self_shielding_length = &H2_self_shielding_length_gr;
    // gr_float isrf                          = 1.7;
    // grackle_fields.isrf_habing             = &isrf;

    solve_chemistry(&code_units_simulation, &grackle_fields, dt / code_units_simulation.time_units);

    rho                      = gr_rho;
    u                        = gr_u;
    HI_fraction              = HI_density / gr_rho;
    HII_fraction             = HII_density / gr_rho;
    HM_fraction              = HM_density / gr_rho;
    HeI_fraction             = HeI_density / gr_rho;
    HeII_fraction            = HeII_density / gr_rho;
    HeIII_fraction           = HeIII_density / gr_rho;
    H2I_fraction             = H2I_density / gr_rho;
    H2II_fraction            = H2II_density / gr_rho;
    DI_fraction              = DI_density / gr_rho;
    DII_fraction             = DII_density / gr_rho;
    HDI_fraction             = HDI_density / gr_rho;
    e_fraction               = e_density / gr_rho;
    metal_fraction           = metal_density / gr_rho;
    volumetric_heating_rate  = volumetric_heating_rate_gr;
    specific_heating_rate    = specific_heating_rate_gr;
    RT_heating_rate          = RT_heating_rate_gr;
    RT_HI_ionization_rate    = RT_HI_ionization_rate_gr;
    RT_HeI_ionization_rate   = RT_HeI_ionization_rate_gr;
    RT_HeII_ionization_rate  = RT_HeII_ionization_rate_gr;
    RT_H2_dissociation_rate  = RT_H2_dissociation_rate_gr;
    H2_self_shielding_length = H2_self_shielding_length_gr;
}

struct grackle_options
{
    std::optional<int>   with_radiative_cooling                 = std::nullopt;
    std::optional<int>   primordial_chemistry                   = std::nullopt;
    std::optional<int>   h2_on_dust                             = std::nullopt;
    std::optional<int>   metal_cooling                          = std::nullopt;
    std::optional<int>   cmb_temperature_floor                  = std::nullopt;
    std::optional<int>   UVbackground                           = std::nullopt;
    std::optional<float> UVbackground_redshift_on               = std::nullopt;
    std::optional<float> UVbackground_redshift_fullon           = std::nullopt;
    std::optional<float> UVbackground_redshift_drop             = std::nullopt;
    std::optional<float> UVbackground_redshift_off              = std::nullopt;
    std::optional<float> Gamma                                  = std::nullopt;
    std::optional<int>   three_body_rate                        = std::nullopt;
    std::optional<int>   cie_cooling                            = std::nullopt;
    std::optional<int>   h2_optical_depth_approximation         = std::nullopt;
    std::optional<int>   photoelectric_heating_rate             = std::nullopt;
    std::optional<int>   Compton_xray_heating                   = std::nullopt;
    std::optional<int>   LWbackground_intensity                 = std::nullopt;
    std::optional<int>   LWbackground_sawtooth_suppression      = std::nullopt;
    std::optional<int>   use_volumetric_heating_rate            = std::nullopt;
    std::optional<int>   use_specific_heating_rate              = std::nullopt;
    std::optional<int>   use_radiative_transfer                 = std::nullopt;
    std::optional<int>   radiative_transfer_coupled_rate_solver = std::nullopt;
    std::optional<int>   radiative_transfer_intermediate_step   = std::nullopt;
    std::optional<int>   radiative_transfer_hydrogen_only       = std::nullopt;
    std::optional<int>   H2_self_shielding                      = std::nullopt;
    std::optional<int>   dust_chemistry                         = std::nullopt;
};

void initGrackle(const std::string grackle_data_file_path, const grackle_options options,
                 const double density_units = 1.67e-24, const double length_units = 1.0, const double time_units = 1e12,
                 const double a_units = 1., const double a_value = 1., const int comoving_coordinates = 0)
{
    grackle_verbose = 0;
    // Units in cgs
    code_units_simulation.density_units        = density_units; // m_sun / (pc * pc * pc);
    code_units_simulation.length_units         = length_units;  // pc;
    code_units_simulation.time_units           = time_units;    // code_time;
    code_units_simulation.velocity_units       = code_units_simulation.length_units / code_units_simulation.time_units;
    code_units_simulation.a_units              = a_units;
    code_units_simulation.a_value              = a_value;
    code_units_simulation.comoving_coordinates = comoving_coordinates;

    chemistry_data_simulation = new chemistry_data;
    set_default_chemistry_parameters(chemistry_data_simulation);

    char* grackle_data_file_path_copy = new char[grackle_data_file_path.size()];
    strncpy(grackle_data_file_path_copy, grackle_data_file_path.c_str(), grackle_data_file_path.size());
    chemistry_data_simulation->grackle_data_file = grackle_data_file_path_copy;

    chemistry_data_simulation->use_grackle = 1;
    auto set_option                        = [](auto& grackle_data_field, const auto& optional)
    {
        if (optional.has_value()) grackle_data_field = optional.value();
    };
    set_option(chemistry_data_simulation->with_radiative_cooling, options.with_radiative_cooling);
    set_option(chemistry_data_simulation->primordial_chemistry, options.primordial_chemistry);
    set_option(chemistry_data_simulation->h2_on_dust, options.h2_on_dust);
    set_option(chemistry_data_simulation->metal_cooling, options.metal_cooling);
    set_option(chemistry_data_simulation->cmb_temperature_floor, options.cmb_temperature_floor);
    set_option(chemistry_data_simulation->UVbackground, options.UVbackground);
    set_option(chemistry_data_simulation->UVbackground_redshift_on, options.UVbackground_redshift_on);
    set_option(chemistry_data_simulation->UVbackground_redshift_fullon, options.UVbackground_redshift_fullon);
    set_option(chemistry_data_simulation->UVbackground_redshift_drop, options.UVbackground_redshift_drop);
    set_option(chemistry_data_simulation->UVbackground_redshift_off, options.UVbackground_redshift_off);
    set_option(chemistry_data_simulation->Gamma, options.Gamma);
    set_option(chemistry_data_simulation->three_body_rate, options.three_body_rate);
    set_option(chemistry_data_simulation->cie_cooling, options.cie_cooling);
    set_option(chemistry_data_simulation->h2_optical_depth_approximation, options.h2_optical_depth_approximation);
    set_option(chemistry_data_simulation->photoelectric_heating_rate, options.photoelectric_heating_rate);
    set_option(chemistry_data_simulation->Compton_xray_heating, options.Compton_xray_heating);
    set_option(chemistry_data_simulation->LWbackground_intensity, options.LWbackground_intensity);
    set_option(chemistry_data_simulation->LWbackground_sawtooth_suppression, options.LWbackground_sawtooth_suppression);
    set_option(chemistry_data_simulation->use_volumetric_heating_rate, options.use_volumetric_heating_rate);
    set_option(chemistry_data_simulation->use_specific_heating_rate, options.use_specific_heating_rate);
    set_option(chemistry_data_simulation->use_radiative_transfer, options.use_radiative_transfer);
    set_option(chemistry_data_simulation->radiative_transfer_coupled_rate_solver,
               options.radiative_transfer_coupled_rate_solver);
    set_option(chemistry_data_simulation->radiative_transfer_intermediate_step,
               options.radiative_transfer_intermediate_step);
    set_option(chemistry_data_simulation->radiative_transfer_hydrogen_only, options.radiative_transfer_hydrogen_only);
    set_option(chemistry_data_simulation->H2_self_shielding, options.H2_self_shielding);
    set_option(chemistry_data_simulation->dust_chemistry, options.dust_chemistry);

    initialize_chemistry_data(&code_units_simulation);
}

void cleanGrackle(void)
{
    _free_chemistry_data(grackle_data, &grackle_rates);
    delete chemistry_data_simulation->grackle_data_file;
    delete chemistry_data_simulation;
}
