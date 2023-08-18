//
// Created by Noah Kubli on 24.11.22.
//

extern "C"
{
#include <grackle.h>
}

#include "cooler.hpp"
#include "cooler_field_data_content.h"

#include <array>
#include <vector>
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
    T m_code_in_ms = 1e16;
    //! @brief code unit length in kpc
    T l_code_in_kpc = 46400.;
    //! @brief Path to Grackle data file
    std::string grackle_data_file_path = CMAKE_SOURCE_DIR "/extern/grackle/grackle_repo/input/CloudyData_UVB=HM2012.h5";

    Impl();

    struct GlobalValues
    {
        code_units             units;
        chemistry_data         data;
        chemistry_data_storage rates;
    };
    GlobalValues global_values;

    constexpr static std::array fieldNames{
        "m_code_in_ms", "l_code_in_kpc",
        // Grackle parameters
        "use_grackle", "with_radiative_cooling", "primordial_chemistry", "dust_chemistry", "metal_cooling",
        "UVbackground",
        //!
        //! d.char *grackle_data_file",
        "cmb_temperature_floor", "Gamma", "h2_on_dust", "use_dust_density_field", "dust_recombination_cooling",
        "photoelectric_heating", "photoelectric_heating_rate", "use_isrf_field", "interstellar_radiation_field",
        "use_volumetric_heating_rate", "use_specific_heating_rate", "three_body_rate", "cie_cooling",
        "h2_optical_depth_approximation", "ih2co", "ipiht", "HydrogenFractionByMass", "DeuteriumToHydrogenRatio",
        "SolarMetalFractionByMass", "local_dust_to_gas_ratio", "NumberOfTemperatureBins", "CaseBRecombination",
        "TemperatureStart", "TemperatureEnd", "NumberOfDustTemperatureBins", "DustTemperatureStart",
        "DustTemperatureEnd", "Compton_xray_heating", "LWbackground_sawtooth_suppression", "LWbackground_intensity",
        "UVbackground_redshift_on", "UVbackground_redshift_off", "UVbackground_redshift_fullon",
        "UVbackground_redshift_drop", "cloudy_electron_fraction_factor", "use_radiative_transfer",
        "radiative_transfer_coupled_rate_solver", "radiative_transfer_intermediate_step",
        "radiative_transfer_hydrogen_only", "self_shielding_method", "H2_self_shielding", "H2_custom_shielding",
        "h2_charge_exchange_rate", "h2_dust_rate", "h2_h_cooling_rate", "collisional_excitation_rates",
        "collisional_ionisation_rates", "recombination_cooling_rates", "bremsstrahlung_cooling_rates"};

    auto fieldsTuple()
    {
        auto& d = global_values.data;
        return std::tie(
            m_code_in_ms, l_code_in_kpc, d.use_grackle, d.with_radiative_cooling, d.primordial_chemistry,
            d.dust_chemistry, d.metal_cooling, d.UVbackground,
            //!
            //! d.char *grackle_data_file,
            d.cmb_temperature_floor, d.Gamma, d.h2_on_dust, d.use_dust_density_field, d.dust_recombination_cooling,
            d.photoelectric_heating, d.photoelectric_heating_rate, d.use_isrf_field, d.interstellar_radiation_field,
            d.use_volumetric_heating_rate, d.use_specific_heating_rate, d.three_body_rate, d.cie_cooling,
            d.h2_optical_depth_approximation, d.ih2co, d.ipiht, d.HydrogenFractionByMass, d.DeuteriumToHydrogenRatio,
            d.SolarMetalFractionByMass, d.local_dust_to_gas_ratio, d.NumberOfTemperatureBins, d.CaseBRecombination,
            d.TemperatureStart, d.TemperatureEnd, d.NumberOfDustTemperatureBins, d.DustTemperatureStart,
            d.DustTemperatureEnd, d.Compton_xray_heating, d.LWbackground_sawtooth_suppression, d.LWbackground_intensity,
            d.UVbackground_redshift_on, d.UVbackground_redshift_off, d.UVbackground_redshift_fullon,
            d.UVbackground_redshift_drop, d.cloudy_electron_fraction_factor, d.use_radiative_transfer,
            d.radiative_transfer_coupled_rate_solver, d.radiative_transfer_intermediate_step,
            d.radiative_transfer_hydrogen_only, d.self_shielding_method, d.H2_self_shielding, d.H2_custom_shielding,
            d.h2_charge_exchange_rate, d.h2_dust_rate, d.h2_h_cooling_rate, d.collisional_excitation_rates,
            d.collisional_ionisation_rates, d.recombination_cooling_rates, d.bremsstrahlung_cooling_rates);
    }
    static_assert(fieldNames.size() == std::tuple_size_v<decltype(((Impl*)nullptr)->fieldsTuple())>);

    std::vector<FieldVariant> getFields()
    {
        return std::apply([](auto&... a) { return std::vector<FieldVariant>{&a...}; }, fieldsTuple());
    }

    static std::vector<const char*> getFieldNames()
    {
        auto a = std::apply([](auto&... a) { return std::array<const char*, sizeof...(a)>{(&a[0])...}; }, fieldNames);
        return std::vector(a.begin(), a.end());
    }

    void init(int comoving_coordinates, std::optional<T> time_unit);

    chemistry_data getDefaultChemistryData()
    {

        chemistry_data data_default;
        local_initialize_chemistry_parameters(&data_default);
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

    T cooling_time(T& rho, T& u, T& HI_fraction, T& HII_fraction, T& HM_fraction, T& HeI_fraction, T& HeII_fraction,
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
void Cooler<T>::init(const int comoving_coordinates, const std::optional<T> time_unit)
{
    impl_ptr->init(comoving_coordinates, time_unit);
}

template<typename T>
std::vector<typename Cooler<T>::FieldVariant> Cooler<T>::getFields()
{
    return impl_ptr->getFields();
}

template<typename T>
std::vector<const char*> Cooler<T>::getFieldNames()
{
    return Cooler<T>::Impl::getFieldNames();
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

template<typename T>
T Cooler<T>::cooling_time(T& rho, T& u, T& HI_fraction, T& HII_fraction, T& HM_fraction, T& HeI_fraction,
                          T& HeII_fraction, T& HeIII_fraction, T& H2I_fraction, T& H2II_fraction, T& DI_fraction,
                          T& DII_fraction, T& HDI_fraction, T& e_fraction, T& metal_fraction,
                          T& volumetric_heating_rate, T& specific_heating_rate, T& RT_heating_rate,
                          T& RT_HI_ionization_rate, T& RT_HeI_ionization_rate, T& RT_HeII_ionization_rate,
                          T& RT_H2_dissociation_rate, T& H2_self_shielding_length)
{
    return impl_ptr->cooling_time(rho, u, HI_fraction, HII_fraction, HM_fraction, HeI_fraction, HeII_fraction,
                                  HeIII_fraction, H2I_fraction, H2II_fraction, DI_fraction, DII_fraction, HDI_fraction,
                                  e_fraction, metal_fraction, volumetric_heating_rate, specific_heating_rate,
                                  RT_heating_rate, RT_HI_ionization_rate, RT_HeI_ionization_rate,
                                  RT_HeII_ionization_rate, RT_H2_dissociation_rate, H2_self_shielding_length);
}

template struct Cooler<double>;
template struct Cooler<float>;

// Implementation of Cooler::Impl
template<typename T>
Cooler<T>::Impl::Impl()
{
    local_initialize_chemistry_parameters(&global_values.data);
    global_values.data.grackle_data_file = &grackle_data_file_path[0];
}

template<typename T>
void Cooler<T>::Impl::init(const int comoving_coordinates, std::optional<T> tu)
{
    grackle_verbose = 1;

    // Density
    const double density_unit = m_code_in_ms * ms_g / std::pow(l_code_in_kpc * kp_cm, 3);
    // Time
    const double time_unit = tu.value_or(std::sqrt(1. / (density_unit * G_newton)));
    // Length
    const double length_unit = l_code_in_kpc * kp_cm;
    // Velocity
    const double velocity_unit = length_unit / time_unit;

    global_values.units.density_units        = density_unit; // m_sun / (pc * pc * pc);
    global_values.units.time_units           = time_unit;    // code_time;
    global_values.units.length_units         = length_unit;  // pc;
    global_values.units.velocity_units       = velocity_unit;
    global_values.units.a_units              = 1.0;
    global_values.units.a_value              = 1.0;
    global_values.units.comoving_coordinates = comoving_coordinates;

#ifndef NDEBUG
    std::cout << "debug\n";
    std::cout << m_code_in_ms << "\t" << ms_g << "\t" << l_code_in_kpc << "\n";
    std::cout << "code units\n";
    std::cout << global_values.units.density_units << "\t" << global_values.units.time_units << "\t"
              << global_values.units.length_units << "\n";

#endif
    global_values.data.grackle_data_file = &grackle_data_file_path[0];
    if (0 == local_initialize_chemistry_data(&global_values.data, &global_values.rates, &global_values.units))
    {
        std::cout << global_values.data.with_radiative_cooling << std::endl;
        throw std::runtime_error("Grackle: Error in _initialize_chemistry_data");
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

template<typename T>
T Cooler<T>::Impl::cooling_time(T& rho, T& u, T& HI_fraction, T& HII_fraction, T& HM_fraction, T& HeI_fraction,
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
    gr_float time(0.0);
    if (0 == local_calculate_cooling_time(&global_values.data, &global_values.rates, &global_values.units,
                                          &grackle_fields.data, &time))
    {
        throw std::runtime_error("Grackle: Error in local_calculate_cooling_time");
    }

    return time;
}

} // namespace cooling
