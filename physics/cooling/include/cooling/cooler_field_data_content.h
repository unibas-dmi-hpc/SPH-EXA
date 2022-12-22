//
// Created by Noah Kubli on 29.11.22.
//

#ifndef SPHEXA_COOLER_FIELD_DATA_CONTENT_H
#define SPHEXA_COOLER_FIELD_DATA_CONTENT_H

extern "C"
{
#include "grackle.h"
};

// Implementation of cooling functions
template<typename T>
struct cooler_field_data_content
{
    grackle_field_data data;
    int                zero[3] = {0, 0, 0};
    int                one[3]  = {1, 1, 1};
    gr_float           gr_rho;
    gr_float           gr_u;
    gr_float           x_velocity;
    gr_float           y_velocity;
    gr_float           z_velocity;
    gr_float           HI_density;
    gr_float           HII_density;
    gr_float           HM_density;
    gr_float           HeI_density;
    gr_float           HeII_density;
    gr_float           HeIII_density;
    gr_float           H2I_density;
    gr_float           H2II_density;
    gr_float           DI_density;
    gr_float           DII_density;
    gr_float           HDI_density;
    gr_float           e_density;
    gr_float           metal_density;
    gr_float           volumetric_heating_rate_gr;
    gr_float           specific_heating_rate_gr;
    gr_float           RT_heating_rate_gr;
    gr_float           RT_HI_ionization_rate_gr;
    gr_float           RT_HeI_ionization_rate_gr;
    gr_float           RT_HeII_ionization_rate_gr;
    gr_float           RT_H2_dissociation_rate_gr;
    gr_float           H2_self_shielding_length_gr;

    void assign_field_data(T& rho, T& u, T& HI_fraction, T& HII_fraction, T& HM_fraction, T& HeI_fraction,
                           T& HeII_fraction, T& HeIII_fraction, T& H2I_fraction, T& H2II_fraction, T& DI_fraction,
                           T& DII_fraction, T& HDI_fraction, T& e_fraction, T& metal_fraction,
                           T& volumetric_heating_rate, T& specific_heating_rate, T& RT_heating_rate,
                           T& RT_HI_ionization_rate, T& RT_HeI_ionization_rate, T& RT_HeII_ionization_rate,
                           T& RT_H2_dissociation_rate, T& H2_self_shielding_length)
    {
        data.grid_rank      = 3;
        data.grid_dimension = one;
        data.grid_start     = zero;
        data.grid_end       = zero;
        data.grid_dx        = 0.0;

        gr_rho                      = (gr_float)rho;
        gr_u                        = (gr_float)u;
        x_velocity                  = 0.;
        y_velocity                  = 0.;
        z_velocity                  = 0.;
        HI_density                  = (gr_float)HI_fraction * (gr_float)rho;
        HII_density                 = (gr_float)HII_fraction * (gr_float)rho;
        HM_density                  = (gr_float)HM_fraction * (gr_float)rho;
        HeI_density                 = (gr_float)HeI_fraction * (gr_float)rho;
        HeII_density                = (gr_float)HeII_fraction * (gr_float)rho;
        HeIII_density               = (gr_float)HeIII_fraction * (gr_float)rho;
        H2I_density                 = (gr_float)H2I_fraction * (gr_float)rho;
        H2II_density                = (gr_float)H2II_fraction * (gr_float)rho;
        DI_density                  = (gr_float)DI_fraction * (gr_float)rho;
        DII_density                 = (gr_float)DII_fraction * (gr_float)rho;
        HDI_density                 = (gr_float)HDI_fraction * (gr_float)rho;
        e_density                   = (gr_float)e_fraction * (gr_float)rho;
        metal_density               = (gr_float)metal_fraction * (gr_float)rho;
        volumetric_heating_rate_gr  = (gr_float)volumetric_heating_rate;
        specific_heating_rate_gr    = (gr_float)specific_heating_rate;
        RT_heating_rate_gr          = (gr_float)RT_heating_rate;
        RT_HI_ionization_rate_gr    = (gr_float)RT_HI_ionization_rate;
        RT_HeI_ionization_rate_gr   = (gr_float)RT_HeI_ionization_rate;
        RT_HeII_ionization_rate_gr  = (gr_float)RT_HeII_ionization_rate;
        RT_H2_dissociation_rate_gr  = (gr_float)RT_H2_dissociation_rate;
        H2_self_shielding_length_gr = (gr_float)H2_self_shielding_length;

        data.density         = &gr_rho;
        data.internal_energy = &gr_u;
        data.x_velocity      = &x_velocity;
        data.y_velocity      = &y_velocity;
        data.z_velocity      = &z_velocity;
        data.HI_density      = &HI_density;
        data.HII_density     = &HII_density;
        data.HeI_density     = &HeI_density;
        data.HeII_density    = &HeII_density;
        data.HeIII_density   = &HeIII_density;
        data.e_density       = &e_density;
        data.HM_density      = &HM_density;
        data.H2I_density     = &H2I_density;
        data.H2II_density    = &H2II_density;
        data.DI_density      = &DI_density;
        data.DII_density     = &DII_density;
        data.HDI_density     = &HDI_density;
        data.metal_density   = &metal_density;

        data.volumetric_heating_rate  = &volumetric_heating_rate_gr;
        data.specific_heating_rate    = &specific_heating_rate_gr;
        data.RT_heating_rate          = &RT_heating_rate_gr;
        data.RT_HI_ionization_rate    = &RT_HI_ionization_rate_gr;
        data.RT_HeI_ionization_rate   = &RT_HeI_ionization_rate_gr;
        data.RT_HeII_ionization_rate  = &RT_HeII_ionization_rate_gr;
        data.RT_H2_dissociation_rate  = &RT_H2_dissociation_rate_gr;
        data.H2_self_shielding_length = &H2_self_shielding_length_gr;
    };

    void get_field_data(T& rho, T& u, T& HI_fraction, T& HII_fraction, T& HM_fraction, T& HeI_fraction,
                        T& HeII_fraction, T& HeIII_fraction, T& H2I_fraction, T& H2II_fraction, T& DI_fraction,
                        T& DII_fraction, T& HDI_fraction, T& e_fraction, T& metal_fraction, T& volumetric_heating_rate,
                        T& specific_heating_rate, T& RT_heating_rate, T& RT_HI_ionization_rate,
                        T& RT_HeI_ionization_rate, T& RT_HeII_ionization_rate, T& RT_H2_dissociation_rate,
                        T& H2_self_shielding_length)
    {
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
};

#endif // SPHEXA_COOLER_FIELD_DATA_CONTENT_H
