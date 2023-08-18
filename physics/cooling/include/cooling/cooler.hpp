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
 * @author Noah Kubli <noah.kubli@uzh.ch>
 */

#pragma once

#include <cstring>
#include <optional>
#include <memory>
#include <map>
#include <any>
#include <variant>
#include <vector>
#include <iostream>

namespace cooling
{

template<typename T>
struct Cooler
{
    Cooler();

    ~Cooler();

    //! @brief Init Cooler. Must be called before any other function is used and after parameters are set
    void init(int comoving_coordinates, std::optional<T> time_unit = std::nullopt);

    //! @brief Calls the GRACKLE library to integrate the cooling and chemistry fields
    void cool_particle(const T& dt, T& rho, T& u, T& HI_fraction, T& HII_fraction, T& HM_fraction, T& HeI_fraction,
                       T& HeII_fraction, T& HeIII_fraction, T& H2I_fraction, T& H2II_fraction, T& DI_fraction,
                       T& DII_fraction, T& HDI_fraction, T& e_fraction, T& metal_fraction, T& volumetric_heating_rate,
                       T& specific_heating_rate, T& RT_heating_rate, T& RT_HI_ionization_rate,
                       T& RT_HeI_ionization_rate, T& RT_HeII_ionization_rate, T& RT_H2_dissociation_rate,
                       T& H2_self_shielding_length);

    //! @brief Calculate the temperature in K (physical units) from the internal energy (code units) and the chemistry
    //! composition
    T energy_to_temperature(const T& dt, T& rho, T& u, T& HI_fraction, T& HII_fraction, T& HM_fraction, T& HeI_fraction,
                            T& HeII_fraction, T& HeIII_fraction, T& H2I_fraction, T& H2II_fraction, T& DI_fraction,
                            T& DII_fraction, T& HDI_fraction, T& e_fraction, T& metal_fraction,
                            T& volumetric_heating_rate, T& specific_heating_rate, T& RT_heating_rate,
                            T& RT_HI_ionization_rate, T& RT_HeI_ionization_rate, T& RT_HeII_ionization_rate,
                            T& RT_H2_dissociation_rate, T& H2_self_shielding_length);

    //! @brief Calculate pressure using the chemistry composition
    T pressure(T& rho, T& u, T& HI_fraction, T& HII_fraction, T& HM_fraction, T& HeI_fraction, T& HeII_fraction,
               T& HeIII_fraction, T& H2I_fraction, T& H2II_fraction, T& DI_fraction, T& DII_fraction, T& HDI_fraction,
               T& e_fraction, T& metal_fraction, T& volumetric_heating_rate, T& specific_heating_rate,
               T& RT_heating_rate, T& RT_HI_ionization_rate, T& RT_HeI_ionization_rate, T& RT_HeII_ionization_rate,
               T& RT_H2_dissociation_rate, T& H2_self_shielding_length);

    //! @brief Calculate adiabatic index from chemistry composition
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

    template<class Archive>
    void loadOrStoreAttributes(Archive* ar)
    {
        auto fieldNames = getFieldNames();
        auto fields     = getFields();
        //! @brief load or store an attribute, skips non-existing attributes on load.
        auto optionalIO = [ar](const std::string& attribute, auto* location, size_t attrSize)
        {
            try
            {
                ar->stepAttribute("cooling::" + attribute, location, attrSize);
            }
            catch (std::out_of_range&)
            {
                std::cout << "Attribute cooling::" << attribute
                          << " not set in file or initializer, setting to default value " << *location << std::endl;
            }
        };
        for (size_t i = 0; i < fieldNames.size(); i++)
        {
            std::visit([&](auto* location) { optionalIO(std::string(fieldNames[i]), location, 1); }, fields[i]);
        }
    }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_ptr;
    using FieldVariant = std::variant<float*, double*, int*>;
    static std::vector<const char*> getFieldNames();
    std::vector<FieldVariant>       getFields();
};
} // namespace cooling
