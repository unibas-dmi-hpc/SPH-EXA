//
// Created by Noah Kubli on 28.02.22.
//

#ifndef SPHEXA_COOLING_HPP
#define SPHEXA_COOLING_HPP

#define CONFIG_BFLOAT_8
extern "C"
{
#include <grackle.h>
}
#include <cmath>
static constexpr double m_sun = 1.989e33;
static constexpr double pc    = 3.086e18;
static constexpr double G_cgs = 6.674e-8;

static code_units grackle_units;

template<typename T, typename Dataset>
void cool_particle(
        T& dt,
        T& rho,
        T& u,
        T& HI_fraction,
        T& HII_fraction,
        T& HeI_fraction,
        T& HeII_fraction,
        T& HeIII_fraction,
        T& e_fraction,
        T& HM_fraction,
        T& H2I_fraction,
        T& H2II_fraction,
        T& DI_fraction,
        T& DII_fraction,
        T& HDI_fraction,
        T& metal_fraction
)
{
    code_units         grackle_units_copy = grackle_units;
    grackle_field_data grackle_fields;
    grackle_fields.grid_rank      = 3;
    int zero[]                    = {0, 0, 0};
    int one[]                     = {1, 1, 1};
    grackle_fields.grid_dimension = one;
    grackle_fields.grid_start     = zero;
    grackle_fields.grid_end       = zero;
    grackle_fields.grid_dx        = 0.0;

    gr_float gr_rho                        = (gr_float)rho;
    grackle_fields.density                 = &gr_rho;
    gr_float gr_u                          = (gr_float)u;
    grackle_fields.internal_energy         = &gr_u;
    gr_float x_velocity                    = 0.;
    grackle_fields.x_velocity              = &x_velocity;
    gr_float y_velocity                    = 0.;
    grackle_fields.y_velocity              = &y_velocity;
    gr_float z_velocity                    = 0.;
    grackle_fields.z_velocity              = &z_velocity;
    gr_float HI_density                    = (gr_float)HI_fraction * (gr_float)rho;
    gr_float HII_density                   = (gr_float)HII_fraction * (gr_float)rho;
    gr_float HeI_density                   = (gr_float)HeI_fraction * (gr_float)rho;
    gr_float HeII_density                  = (gr_float)HeII_fraction * (gr_float)rho;
    gr_float HeIII_density                 = (gr_float)HeIII_fraction * (gr_float)rho;
    gr_float e_density                     = (gr_float)e_fraction * (gr_float)rho;
    gr_float HM_density                    = (gr_float)HM_fraction * (gr_float)rho;
    gr_float H2I_density                   = (gr_float)H2I_fraction * (gr_float)rho;
    gr_float H2II_density                  = (gr_float)H2II_fraction * (gr_float)rho;
    gr_float DI_density                    = (gr_float)DI_fraction * (gr_float)rho;
    gr_float DII_density                   = (gr_float)DII_fraction * (gr_float)rho;
    gr_float HDI_density                   = (gr_float)HDI_fraction * (gr_float)rho;
    gr_float metal_density                 = (gr_float)metal_fraction * (gr_float)rho;
    grackle_fields.HI_density              = &HI_density;
    grackle_fields.HII_density             = &HII_density;
    grackle_fields.HeI_density             = &HeI_density;
    grackle_fields.HeII_density            = &HeII_density;
    grackle_fields.HeIII_density           = &HeIII_density;
    grackle_fields.e_density               = &e_density;
    grackle_fields.HM_density              = &HM_density;
    grackle_fields.H2I_density             = &H2I_density;
    grackle_fields.H2II_density            = &H2II_density;
    grackle_fields.DI_density              = &DI_density;
    grackle_fields.DII_density             = &DII_density;
    grackle_fields.HDI_density             = &HDI_density;
    grackle_fields.metal_density           = &metal_density;
    gr_float vhr                           = 0.;
    gr_float shr                           = 0.;
    gr_float HI_ir                         = 0.;
    gr_float HeI_ir                        = 0.;
    gr_float HeII_ir                       = 0.;
    gr_float H2_dr                         = 0.;
    gr_float hr                            = 0.;
    grackle_fields.volumetric_heating_rate = &vhr;
    grackle_fields.specific_heating_rate   = &shr;
    grackle_fields.RT_HI_ionization_rate   = &HI_ir;
    grackle_fields.RT_HeI_ionization_rate  = &HeI_ir;
    grackle_fields.RT_HeII_ionization_rate = &HeII_ir;
    grackle_fields.RT_H2_dissociation_rate = &H2_dr;
    grackle_fields.RT_heating_rate         = &hr;
    gr_float isrf                          = 1.7;
    grackle_fields.isrf_habing             = &isrf;

    // std::cout << grackle_units_copy.density_units << std::endl;
    // std::cout << dt[i] / grackle_units.time_units << std::endl;
    // std::cout << HI_density / gr_rho << std::endl;
    // std::cout << HeI_density / gr_rho << std::endl;
    // std::cout << gr_u << std::endl;

    solve_chemistry(&grackle_units_copy, &grackle_fields, dt / grackle_units.time_units);
    // std::cout << HI_density/gr_rho << std::endl;
    rho            = gr_rho;
    HI_fraction    = HI_density / gr_rho;
    HII_fraction   = HII_density / gr_rho;
    HeI_fraction   = HeI_density / gr_rho;
    HeII_fraction  = HeII_density / gr_rho;
    HeIII_fraction = HeIII_density / gr_rho;
    e_fraction     = e_density / gr_rho;
    HM_fraction    = HM_density / gr_rho;
    H2I_fraction   = H2I_density / gr_rho;
    H2II_fraction  = H2II_density / gr_rho;
    DI_fraction    = DI_density / gr_rho;
    DII_fraction   = DII_density / gr_rho;
    HDI_fraction   = HDI_density / gr_rho;
    metal_fraction = metal_density / gr_rho;
    u              = gr_u;
}

void initGrackle(void)
{
    grackle_verbose = 0;
    // Units in cgs

    // const double code_time = std::sqrt(pc*pc*pc / m_sun / G_cgs * 4. * M_PI * M_PI);

    grackle_units.density_units        = 1.67e-24; // m_sun / (pc * pc * pc);
    grackle_units.length_units         = 1.0;      // pc;
    grackle_units.time_units           = 1e12;     // code_time;
    grackle_units.velocity_units       = grackle_units.length_units / grackle_units.time_units;
    grackle_units.a_units              = 1.;
    grackle_units.a_value              = 1.;
    grackle_units.comoving_coordinates = 0;

    chemistry_data* grackle_data = new chemistry_data;
    set_default_chemistry_parameters(grackle_data);
    grackle_data->use_grackle            = 1;
    grackle_data->use_isrf_field         = 1;
    grackle_data->with_radiative_cooling = 1;
    grackle_data->primordial_chemistry   = 3;
    grackle_data->dust_chemistry         = 1;
    grackle_data->metal_cooling          = 1;
    grackle_data->UVbackground           = 1;
    static char data_file[] = "grackle_data_files/input/CloudyData_UVB=HM2012.h5";
    grackle_data->grackle_data_file = data_file;
    initialize_chemistry_data(&grackle_units);
}
void cleanGrackle(void) { _free_chemistry_data(grackle_data, &grackle_rates); }
#endif // SPHEXA_COOLING_HPP
