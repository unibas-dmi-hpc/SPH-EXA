//
// Created by Noah Kubli on 29.11.22.
//

#pragma once

extern "C"
{
#include "grackle.h"
}

#include "cooler.hpp"

template<typename T>
struct cooler_field_data_content
{
    using ParticleType = typename cooling::Cooler<T>::ParticleType;

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

    void assign_field_data(T rho, T u, const ParticleType& particle)
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
        HI_density                  = (gr_float)*std::get<0>(particle) * (gr_float)rho;
        HII_density                 = (gr_float)*std::get<1>(particle) * (gr_float)rho;
        HM_density                  = (gr_float)*std::get<2>(particle) * (gr_float)rho;
        HeI_density                 = (gr_float)*std::get<3>(particle) * (gr_float)rho;
        HeII_density                = (gr_float)*std::get<4>(particle) * (gr_float)rho;
        HeIII_density               = (gr_float)*std::get<5>(particle) * (gr_float)rho;
        H2I_density                 = (gr_float)*std::get<6>(particle) * (gr_float)rho;
        H2II_density                = (gr_float)*std::get<7>(particle) * (gr_float)rho;
        DI_density                  = (gr_float)*std::get<8>(particle) * (gr_float)rho;
        DII_density                 = (gr_float)*std::get<9>(particle) * (gr_float)rho;
        HDI_density                 = (gr_float)*std::get<10>(particle) * (gr_float)rho;
        e_density                   = (gr_float)*std::get<11>(particle) * (gr_float)rho;
        metal_density               = (gr_float)*std::get<12>(particle) * (gr_float)rho;
        volumetric_heating_rate_gr  = (gr_float)*std::get<13>(particle);
        specific_heating_rate_gr    = (gr_float)*std::get<14>(particle);
        RT_heating_rate_gr          = (gr_float)*std::get<15>(particle);
        RT_HI_ionization_rate_gr    = (gr_float)*std::get<16>(particle);
        RT_HeI_ionization_rate_gr   = (gr_float)*std::get<17>(particle);
        RT_HeII_ionization_rate_gr  = (gr_float)*std::get<18>(particle);
        RT_H2_dissociation_rate_gr  = (gr_float)*std::get<19>(particle);
        H2_self_shielding_length_gr = (gr_float)*std::get<20>(particle);

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

    void get_field_data(T& rho, T& u, const ParticleType& particle)
    {
        rho                                                  = gr_rho;
        u                                                    = gr_u;
        *std::get<0>(particle) /*HI_fraction             */  = HI_density / gr_rho;
        *std::get<1>(particle) /*HII_fraction            */  = HII_density / gr_rho;
        *std::get<2>(particle) /*HM_fraction             */  = HM_density / gr_rho;
        *std::get<3>(particle) /*HeI_fraction            */  = HeI_density / gr_rho;
        *std::get<4>(particle) /*HeII_fraction           */  = HeII_density / gr_rho;
        *std::get<5>(particle) /*HeIII_fraction          */  = HeIII_density / gr_rho;
        *std::get<6>(particle) /*H2I_fraction            */  = H2I_density / gr_rho;
        *std::get<7>(particle) /*H2II_fraction           */  = H2II_density / gr_rho;
        *std::get<8>(particle) /*DI_fraction             */  = DI_density / gr_rho;
        *std::get<9>(particle) /*DII_fraction            */  = DII_density / gr_rho;
        *std::get<10>(particle) /*HDI_fraction            */ = HDI_density / gr_rho;
        *std::get<11>(particle) /*e_fraction              */ = e_density / gr_rho;
        *std::get<12>(particle) /*metal_fraction          */ = metal_density / gr_rho;
        *std::get<13>(particle) /*volumetric_heating_rate */ = volumetric_heating_rate_gr;
        *std::get<14>(particle) /*specific_heating_rate   */ = specific_heating_rate_gr;
        *std::get<15>(particle) /*RT_heating_rate         */ = RT_heating_rate_gr;
        *std::get<16>(particle) /*RT_HI_ionization_rate   */ = RT_HI_ionization_rate_gr;
        *std::get<17>(particle) /*RT_HeI_ionization_rate  */ = RT_HeI_ionization_rate_gr;
        *std::get<18>(particle) /*RT_HeII_ionization_rate */ = RT_HeII_ionization_rate_gr;
        *std::get<19>(particle) /*RT_H2_dissociation_rate */ = RT_H2_dissociation_rate_gr;
        *std::get<20>(particle) /*H2_self_shielding_length*/ = H2_self_shielding_length_gr;
    }
};
