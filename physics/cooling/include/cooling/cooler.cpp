//
// Created by Noah Kubli on 24.11.22.
//

extern "C"
{
#include <grackle.h>
}

#include "cooler.hpp"
#include "cooler_field_data_content.h"

#include <map>
#include <any>
#include <cmath>
#include <iostream>

namespace cooling
{

template<typename T>
struct Cooler<T>::Impl
{
    friend struct Cooler<T>;

private:
    //! @brief Solar mass in g
    constexpr static T ms_g = 1.989e33;
    //! @brief kpc in cm
    constexpr static T kp_cm = 3.086e21;
    //! @brief Gravitational constant in cgs units
    constexpr static T G_newton = 6.674e-8;
    //! @brief code unit mass in solar masses
    T ms = 1e16;
    //! @brief code unit length in kpc
    T kpc = 46400.;
    //! @brief Path to Grackle data file
    std::string grackle_data_file_path = CMAKE_SOURCE_DIR "/extern/grackle/grackle_repo/input/CloudyData_UVB=HM2012.h5";

    void initOptions(const std::map<std::string, std::any>& grackle_options);

    // void initOptions(const std::string &grackle_options_file_path);
    struct GlobalValues
    {
        code_units             units;
        chemistry_data         data;
        chemistry_data_storage rates;
    };
    GlobalValues global_values;

    void init(const double ms_sim, const double kp_sim, const int comoving_coordinates,
              const std::optional<std::map<std::string, std::any>> grackleOptions = std::nullopt,
              const std::optional<double>                          t_sim          = std::nullopt);

    chemistry_data getDefaultChemistryData()
    {
        chemistry_data data_default    = _set_default_chemistry_parameters();
        data_default.grackle_data_file = &grackle_data_file_path[0];
        return data_default;
    }

    void cool_particle(const T& dt, T& rho, T& u, T& HI_fraction, T& HII_fraction, T& HM_fraction, T& HeI_fraction,
                       T& HeII_fraction, T& HeIII_fraction, T& H2I_fraction, T& H2II_fraction, T& DI_fraction,
                       T& DII_fraction, T& HDI_fraction, T& e_fraction, T& metal_fraction, T& volumetric_heating_rate,
                       T& specific_heating_rate, T& RT_heating_rate, T& RT_HI_ionization_rate,
                       T& RT_HeI_ionization_rate, T& RT_HeII_ionization_rate, T& RT_H2_dissociation_rate,
                       T& H2_self_shielding_length);

    T energy_to_temperature(const T& dt, T& rho, T& u, T& HI_fraction, T& HII_fraction, T& HM_fraction, T& HeI_fraction,
                            T& HeII_fraction, T& HeIII_fraction, T& H2I_fraction, T& H2II_fraction, T& DI_fraction,
                            T& DII_fraction, T& HDI_fraction, T& e_fraction, T& metal_fraction,
                            T& volumetric_heating_rate, T& specific_heating_rate, T& RT_heating_rate,
                            T& RT_HI_ionization_rate, T& RT_HeI_ionization_rate, T& RT_HeII_ionization_rate,
                            T& RT_H2_dissociation_rate, T& H2_self_shielding_length);
};

template<typename T>
void Cooler<T>::Impl::cool_particle(const T& dt, T& rho, T& u, T& HI_fraction, T& HII_fraction, T& HM_fraction,
                                    T& HeI_fraction, T& HeII_fraction, T& HeIII_fraction, T& H2I_fraction,
                                    T& H2II_fraction, T& DI_fraction, T& DII_fraction, T& HDI_fraction, T& e_fraction,
                                    T& metal_fraction, T& volumetric_heating_rate, T& specific_heating_rate,
                                    T& RT_heating_rate, T& RT_HI_ionization_rate, T& RT_HeI_ionization_rate,
                                    T& RT_HeII_ionization_rate, T& RT_H2_dissociation_rate, T& H2_self_shielding_length)
{
    cooler_field_data_content<T> grackle_fields;
    grackle_fields.assign_field_data(
        rho, u, HI_fraction, HII_fraction, HM_fraction, HeI_fraction, HeII_fraction, HeIII_fraction, H2I_fraction,
        H2II_fraction, DI_fraction, DII_fraction, HDI_fraction, e_fraction, metal_fraction, volumetric_heating_rate,
        specific_heating_rate, RT_heating_rate, RT_HI_ionization_rate, RT_HeI_ionization_rate, RT_HeII_ionization_rate,
        RT_H2_dissociation_rate, H2_self_shielding_length);

    // Grackle uses 0 as a return code to indicate failure
    if (0 == local_solve_chemistry(&global_values.data, &global_values.rates, &global_values.units,
                                   &grackle_fields.data, dt))
    {
        throw std::runtime_error("Grackle: Error in local_solve_chemistry");
    }
    grackle_fields.get_field_data(rho, u, HI_fraction, HII_fraction, HM_fraction, HeI_fraction, HeII_fraction,
                                  HeIII_fraction, H2I_fraction, H2II_fraction, DI_fraction, DII_fraction, HDI_fraction,
                                  e_fraction, metal_fraction, volumetric_heating_rate, specific_heating_rate,
                                  RT_heating_rate, RT_HI_ionization_rate, RT_HeI_ionization_rate,
                                  RT_HeII_ionization_rate, RT_H2_dissociation_rate, H2_self_shielding_length);
}

template<typename T>
T Cooler<T>::Impl::energy_to_temperature(const T& dt, T& rho, T& u, T& HI_fraction, T& HII_fraction, T& HM_fraction,
                                         T& HeI_fraction, T& HeII_fraction, T& HeIII_fraction, T& H2I_fraction,
                                         T& H2II_fraction, T& DI_fraction, T& DII_fraction, T& HDI_fraction,
                                         T& e_fraction, T& metal_fraction, T& volumetric_heating_rate,
                                         T& specific_heating_rate, T& RT_heating_rate, T& RT_HI_ionization_rate,
                                         T& RT_HeI_ionization_rate, T& RT_HeII_ionization_rate,
                                         T& RT_H2_dissociation_rate, T& H2_self_shielding_length)
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

    gr_float temp;

    if (0 ==
        local_calculate_temperature(&global_values.data, &global_values.rates, &global_values.units, &grackle_fields,
                                    &temp) == 0)
    {
        throw std::runtime_error("Grackle: Error in local_calculate_temperature");
    }
    return temp;
}

// Implementation of Cooler
template<typename T>
Cooler<T>::Cooler()
    : impl_ptr(new Impl)
{
}

template<typename T>
Cooler<T>::~Cooler() = default;

template<typename T>
void Cooler<T>::init(const double ms_sim, const double kp_sim, const int comoving_coordinates,
                     const std::optional<std::map<std::string, std::any>> grackleOptions,
                     const std::optional<double>                          t_sim)
{
    impl_ptr->init(ms_sim, kp_sim, comoving_coordinates, grackleOptions, t_sim);
}

template<typename T>
void Cooler<T>::cool_particle(const T& dt, T& rho, T& u, T& HI_fraction, T& HII_fraction, T& HM_fraction,
                              T& HeI_fraction, T& HeII_fraction, T& HeIII_fraction, T& H2I_fraction, T& H2II_fraction,
                              T& DI_fraction, T& DII_fraction, T& HDI_fraction, T& e_fraction, T& metal_fraction,
                              T& volumetric_heating_rate, T& specific_heating_rate, T& RT_heating_rate,
                              T& RT_HI_ionization_rate, T& RT_HeI_ionization_rate, T& RT_HeII_ionization_rate,
                              T& RT_H2_dissociation_rate, T& H2_self_shielding_length)
{
    impl_ptr->cool_particle(dt, rho, u, HI_fraction, HII_fraction, HM_fraction, HeI_fraction, HeII_fraction,
                            HeIII_fraction, H2I_fraction, H2II_fraction, DI_fraction, DII_fraction, HDI_fraction,
                            e_fraction, metal_fraction, volumetric_heating_rate, specific_heating_rate, RT_heating_rate,
                            RT_HI_ionization_rate, RT_HeI_ionization_rate, RT_HeII_ionization_rate,
                            RT_H2_dissociation_rate, H2_self_shielding_length);
}

template<typename T>
T Cooler<T>::energy_to_temperature(const T& dt, T& rho, T& u, T& HI_fraction, T& HII_fraction, T& HM_fraction,
                                   T& HeI_fraction, T& HeII_fraction, T& HeIII_fraction, T& H2I_fraction,
                                   T& H2II_fraction, T& DI_fraction, T& DII_fraction, T& HDI_fraction, T& e_fraction,
                                   T& metal_fraction, T& volumetric_heating_rate, T& specific_heating_rate,
                                   T& RT_heating_rate, T& RT_HI_ionization_rate, T& RT_HeI_ionization_rate,
                                   T& RT_HeII_ionization_rate, T& RT_H2_dissociation_rate, T& H2_self_shielding_length)
{
    return impl_ptr->energy_to_temperature(
        dt, rho, u, HI_fraction, HII_fraction, HM_fraction, HeI_fraction, HeII_fraction, HeIII_fraction, H2I_fraction,
        H2II_fraction, DI_fraction, DII_fraction, HDI_fraction, e_fraction, metal_fraction, volumetric_heating_rate,
        specific_heating_rate, RT_heating_rate, RT_HI_ionization_rate, RT_HeI_ionization_rate, RT_HeII_ionization_rate,
        RT_H2_dissociation_rate, H2_self_shielding_length);
}

template struct Cooler<double>;
template struct Cooler<float>;

// Implementation of Cooler::Impl
template<typename T>
void Cooler<T>::Impl::init(const double ms_sim, const double kp_sim, const int comoving_coordinates,
                           const std::optional<std::map<std::string, std::any>> grackleOptions,
                           const std::optional<double>                          t_sim)
{
    ms              = ms_sim;
    kpc             = kp_sim;
    grackle_verbose = 1;

    // Density
    const double density_unit = ms * ms_g / std::pow(kpc * kp_cm, 3);
    // Time
    const double time_unit = t_sim.value_or(std::sqrt(1. / (density_unit * G_newton)));
    // Length
    const double length_unit = kpc * kp_cm;
    // Velocity
    const double velocity_unit = length_unit / time_unit;

    global_values.units.density_units        = density_unit; // m_sun / (pc * pc * pc);
    global_values.units.time_units           = time_unit;    // code_time;
    global_values.units.length_units         = length_unit;  // pc;
    global_values.units.velocity_units       = velocity_unit;
    global_values.units.a_units              = 1.0;
    global_values.units.a_value              = 1.0;
    global_values.units.comoving_coordinates = comoving_coordinates;

    std::cout << "debug\n";
    std::cout << ms << "\t" << ms_g << "\t" << kpc << "\n";
    std::cout << "code units\n";
    std::cout << global_values.units.density_units << "\t" << global_values.units.time_units << "\t"
              << global_values.units.length_units << "\n";

    global_values.data = _set_default_chemistry_parameters();

    global_values.data.grackle_data_file = &grackle_data_file_path[0];

    /*if (grackleOptions.has_value() && grackleOptionsFile.has_value())
        throw std::runtime_error("Specify only one; either grackleOptions or grackleOptionsFile");*/

    if (grackleOptions.has_value()) { initOptions(grackleOptions.value()); }
    // else if (grackleOptionsFile.has_value()) { initOptions(grackleOptionsFile.value()); }
    else
    {
        global_values.data     = getDefaultChemistryData();
        grackle_data_file_path = std::string(global_values.data.grackle_data_file);
        std::cout << grackle_data_file_path << std::endl;
        global_values.data.grackle_data_file = &grackle_data_file_path[0];
    }

    if (0 == _initialize_chemistry_data(&global_values.data, &global_values.rates, &global_values.units))
    {
        std::cout << global_values.data.with_radiative_cooling << std::endl;
        throw std::runtime_error("Grackle: Error in _initialize_chemistry_data");
    }
}

template<typename T>
void Cooler<T>::Impl::initOptions(const std::map<std::string, std::any>& grackle_options)
{
    for (auto [key, value] : grackle_options)
    {
        try
        {
            if (key == "grackle_data_file_path")
            {
                grackle_data_file_path = std::any_cast<std::string>(value);
                global_values.data.grackle_data_file;
            }
            if (key == ("use_grackle")) global_values.data.use_grackle = std::any_cast<int>(value);
            if (key == ("with_radiative_cooling"))
                global_values.data.with_radiative_cooling = std::any_cast<int>(value);
            if (key == ("primordial_chemistry")) global_values.data.primordial_chemistry = std::any_cast<int>(value);
            if (key == ("h2_on_dust")) global_values.data.h2_on_dust = std::any_cast<int>(value);
            if (key == ("metal_cooling")) global_values.data.metal_cooling = std::any_cast<int>(value);
            if (key == ("cmb_temperature_floor")) global_values.data.cmb_temperature_floor = std::any_cast<int>(value);
            if (key == ("UVbackground")) global_values.data.UVbackground = std::any_cast<int>(value);
            if (key == ("UVbackground_redshift_on"))
                global_values.data.UVbackground_redshift_on = std::any_cast<int>(value);
            if (key == ("UVbackground_redshift_fullon"))
                global_values.data.UVbackground_redshift_fullon = std::any_cast<int>(value);
            if (key == ("UVbackground_redshift_drop"))
                global_values.data.UVbackground_redshift_drop = std::any_cast<int>(value);
            if (key == ("UVbackground_redshift_off"))
                global_values.data.UVbackground_redshift_off = std::any_cast<int>(value);
            if (key == ("Gamma")) global_values.data.Gamma = std::any_cast<int>(value);
            if (key == ("three_body_rate")) global_values.data.three_body_rate = std::any_cast<int>(value);
            if (key == ("cie_cooling")) global_values.data.cie_cooling = std::any_cast<int>(value);
            if (key == ("h2_optical_depth_approximation"))
                global_values.data.h2_optical_depth_approximation = std::any_cast<int>(value);
            if (key == ("photoelectric_heating_rate"))
                global_values.data.photoelectric_heating_rate = std::any_cast<int>(value);
            if (key == ("Compton_xray_heating")) global_values.data.Compton_xray_heating = std::any_cast<int>(value);
            if (key == ("LWbackground_intensity"))
                global_values.data.LWbackground_intensity = std::any_cast<int>(value);
            if (key == ("LWbackground_sawtooth_suppression"))
                global_values.data.LWbackground_sawtooth_suppression = std::any_cast<int>(value);
            if (key == ("use_volumetric_heating_rate"))
                global_values.data.use_volumetric_heating_rate = std::any_cast<int>(value);
            if (key == ("use_specific_heating_rate"))
                global_values.data.use_specific_heating_rate = std::any_cast<int>(value);
            if (key == ("use_radiative_transfer"))
                global_values.data.use_radiative_transfer = std::any_cast<int>(value);
            if (key == ("radiative_transfer_coupled_rate_solver"))
                global_values.data.radiative_transfer_coupled_rate_solver = std::any_cast<int>(value);
            if (key == ("radiative_transfer_intermediate_step"))
                global_values.data.radiative_transfer_intermediate_step = std::any_cast<int>(value);
            if (key == ("radiative_transfer_hydrogen_only"))
                global_values.data.radiative_transfer_hydrogen_only = std::any_cast<int>(value);
            if (key == ("H2_self_shielding")) global_values.data.H2_self_shielding = std::any_cast<int>(value);
            if (key == ("dust_chemistry")) global_values.data.dust_chemistry = std::any_cast<int>(value);
        }
        catch (std::bad_any_cast& e)
        {
            std::cout << "Wrong datatype " << key << std::endl;
            std::cout << e.what();
        }
    }
}
} // namespace cooling