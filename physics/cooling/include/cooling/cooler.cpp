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

    T pressure(T& rho, T& u, T& HI_fraction, T& HII_fraction, T& HM_fraction, T& HeI_fraction, T& HeII_fraction,
               T& HeIII_fraction, T& H2I_fraction, T& H2II_fraction, T& DI_fraction, T& DII_fraction, T& HDI_fraction,
               T& e_fraction, T& metal_fraction, T& volumetric_heating_rate, T& specific_heating_rate,
               T& RT_heating_rate, T& RT_HI_ionization_rate, T& RT_HeI_ionization_rate, T& RT_HeII_ionization_rate,
               T& RT_H2_dissociation_rate, T& H2_self_shielding_length);

    T adiabatic_index(T& rho, T& u, T& HI_fraction, T& HII_fraction, T& HM_fraction, T& HeI_fraction, T& HeII_fraction,
                      T& HeIII_fraction, T& H2I_fraction, T& H2II_fraction, T& DI_fraction, T& DII_fraction,
                      T& HDI_fraction, T& e_fraction, T& metal_fraction, T& volumetric_heating_rate,
                      T& specific_heating_rate, T& RT_heating_rate, T& RT_HI_ionization_rate, T& RT_HeI_ionization_rate,
                      T& RT_HeII_ionization_rate, T& RT_H2_dissociation_rate, T& H2_self_shielding_length);
};

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

template<typename T>
T Cooler<T>::pressure(T& rho, T& u, T& HI_fraction, T& HII_fraction, T& HM_fraction, T& HeI_fraction, T& HeII_fraction,
                      T& HeIII_fraction, T& H2I_fraction, T& H2II_fraction, T& DI_fraction, T& DII_fraction,
                      T& HDI_fraction, T& e_fraction, T& metal_fraction, T& volumetric_heating_rate,
                      T& specific_heating_rate, T& RT_heating_rate, T& RT_HI_ionization_rate, T& RT_HeI_ionization_rate,
                      T& RT_HeII_ionization_rate, T& RT_H2_dissociation_rate, T& H2_self_shielding_length)
{
    return impl_ptr->pressure(rho, u, HI_fraction, HII_fraction, HM_fraction, HeI_fraction, HeII_fraction,
                              HeIII_fraction, H2I_fraction, H2II_fraction, DI_fraction, DII_fraction, HDI_fraction,
                              e_fraction, metal_fraction, volumetric_heating_rate, specific_heating_rate,
                              RT_heating_rate, RT_HI_ionization_rate, RT_HeI_ionization_rate, RT_HeII_ionization_rate,
                              RT_H2_dissociation_rate, H2_self_shielding_length);
}

template<typename T>
T Cooler<T>::adiabatic_index(T& rho, T& u, T& HI_fraction, T& HII_fraction, T& HM_fraction, T& HeI_fraction,
                             T& HeII_fraction, T& HeIII_fraction, T& H2I_fraction, T& H2II_fraction, T& DI_fraction,
                             T& DII_fraction, T& HDI_fraction, T& e_fraction, T& metal_fraction,
                             T& volumetric_heating_rate, T& specific_heating_rate, T& RT_heating_rate,
                             T& RT_HI_ionization_rate, T& RT_HeI_ionization_rate, T& RT_HeII_ionization_rate,
                             T& RT_H2_dissociation_rate, T& H2_self_shielding_length)
{
    return impl_ptr->adiabatic_index(
        rho, u, HI_fraction, HII_fraction, HM_fraction, HeI_fraction, HeII_fraction, HeIII_fraction, H2I_fraction,
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

    if (grackleOptions.has_value()) { initOptions(grackleOptions.value()); }
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
    cooler_field_data_content<T> grackle_fields;
    grackle_fields.assign_field_data(
        rho, u, HI_fraction, HII_fraction, HM_fraction, HeI_fraction, HeII_fraction, HeIII_fraction, H2I_fraction,
        H2II_fraction, DI_fraction, DII_fraction, HDI_fraction, e_fraction, metal_fraction, volumetric_heating_rate,
        specific_heating_rate, RT_heating_rate, RT_HI_ionization_rate, RT_HeI_ionization_rate, RT_HeII_ionization_rate,
        RT_H2_dissociation_rate, H2_self_shielding_length);
    gr_float temp;

    if (0 == local_calculate_temperature(&global_values.data, &global_values.rates, &global_values.units,
                                         &grackle_fields.data, &temp))
    {
        throw std::runtime_error("Grackle: Error in local_calculate_temperature");
    }
    return temp;
}

template<typename T>
T Cooler<T>::Impl::pressure(T& rho, T& u, T& HI_fraction, T& HII_fraction, T& HM_fraction, T& HeI_fraction,
                            T& HeII_fraction, T& HeIII_fraction, T& H2I_fraction, T& H2II_fraction, T& DI_fraction,
                            T& DII_fraction, T& HDI_fraction, T& e_fraction, T& metal_fraction,
                            T& volumetric_heating_rate, T& specific_heating_rate, T& RT_heating_rate,
                            T& RT_HI_ionization_rate, T& RT_HeI_ionization_rate, T& RT_HeII_ionization_rate,
                            T& RT_H2_dissociation_rate, T& H2_self_shielding_length)
{
    cooler_field_data_content<T> grackle_fields;
    grackle_fields.assign_field_data(
        rho, u, HI_fraction, HII_fraction, HM_fraction, HeI_fraction, HeII_fraction, HeIII_fraction, H2I_fraction,
        H2II_fraction, DI_fraction, DII_fraction, HDI_fraction, e_fraction, metal_fraction, volumetric_heating_rate,
        specific_heating_rate, RT_heating_rate, RT_HI_ionization_rate, RT_HeI_ionization_rate, RT_HeII_ionization_rate,
        RT_H2_dissociation_rate, H2_self_shielding_length);
    gr_float pressure(0);
    if (0 == local_calculate_pressure(&global_values.data, &global_values.rates, &global_values.units,
                                      &grackle_fields.data, &pressure))
    {
        throw std::runtime_error("Grackle: Error in local_calculate_pressure");
    }
    return T(pressure);
}

template<typename T>
T Cooler<T>::Impl::adiabatic_index(T& rho, T& u, T& HI_fraction, T& HII_fraction, T& HM_fraction, T& HeI_fraction,
                                   T& HeII_fraction, T& HeIII_fraction, T& H2I_fraction, T& H2II_fraction,
                                   T& DI_fraction, T& DII_fraction, T& HDI_fraction, T& e_fraction, T& metal_fraction,
                                   T& volumetric_heating_rate, T& specific_heating_rate, T& RT_heating_rate,
                                   T& RT_HI_ionization_rate, T& RT_HeI_ionization_rate, T& RT_HeII_ionization_rate,
                                   T& RT_H2_dissociation_rate, T& H2_self_shielding_length)
{
    cooler_field_data_content<T> grackle_fields;
    grackle_fields.assign_field_data(
        rho, u, HI_fraction, HII_fraction, HM_fraction, HeI_fraction, HeII_fraction, HeIII_fraction, H2I_fraction,
        H2II_fraction, DI_fraction, DII_fraction, HDI_fraction, e_fraction, metal_fraction, volumetric_heating_rate,
        specific_heating_rate, RT_heating_rate, RT_HI_ionization_rate, RT_HeI_ionization_rate, RT_HeII_ionization_rate,
        RT_H2_dissociation_rate, H2_self_shielding_length);
    gr_float gamma(0);
    local_calculate_gamma(&global_values.data, &global_values.rates, &global_values.units, &grackle_fields.data,
                          &gamma);
    return gamma;
}
} // namespace cooling
