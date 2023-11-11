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

#include <array>
#include <cstring>
#include <optional>
#include <memory>
#include <map>
#include <variant>
#include <vector>
#include <iostream>

#include "cstone/util/type_list.hpp"
#include "cstone/util/value_list.hpp"

namespace cooling
{

template<typename T>
struct Cooler
{
    //! Grackle field names
    inline static constexpr std::array fieldNames{"HI_fraction",
                                                  "HII_fraction",
                                                  "HM_fraction",
                                                  "HeI_fraction",
                                                  "HeII_fraction",
                                                  "HeIII_fraction",
                                                  "H2I_fraction",
                                                  "H2II_fraction",
                                                  "DI_fraction",
                                                  "DII_fraction",
                                                  "HDI_fraction",
                                                  "e_fraction",
                                                  "metal_fraction",
                                                  "volumetric_heating_rate",
                                                  "specific_heating_rate",
                                                  "RT_heating_rate",
                                                  "RT_HI_ionization_rate",
                                                  "RT_HeI_ionization_rate",
                                                  "RT_HeII_ionization_rate",
                                                  "RT_H2_dissociation_rate",
                                                  "H2_self_shielding_length"};

    inline static constexpr size_t numFields = fieldNames.size();

    using CoolingFields = typename util::MakeFieldList<Cooler<T>>::Fields;
    using ParticleType  = util::Reduce<std::tuple, util::Repeat<util::TypeList<std::add_pointer_t<T>>, numFields>>;

    Cooler();

    ~Cooler();

    //! @brief Init Cooler. Must be called before any other function is used and after parameters are set
    void init(int comoving_coordinates, std::optional<T> time_unit = std::nullopt);

    //! @brief Calls the GRACKLE library to integrate the cooling and chemistry fields
    void cool_particle(T dt, T& rho, T& u, const ParticleType& particle);

    //! @brief Calculate the temperature in K (physical units) from the internal energy (code units) and the chemistry
    //! composition
    T energy_to_temperature(T dt, T rho, T u, const ParticleType& particle);

    //! @brief Calculate pressure using the chemistry composition
    T pressure(T rho, T u, const ParticleType& particle);

    //! @brief Calculate adiabatic index from chemistry composition
    T adiabatic_index(T rho, T u, const ParticleType& particle);

    T cooling_time(T rho, T u, const ParticleType& particle);

    // Parameter for cooling time criterion
    T ct_crit{0.1};

    template<class Archive>
    void loadOrStoreAttributes(Archive* ar)
    {
        auto parameterNames = getParameterNames();
        auto parameters     = getParameters();
        //! @brief load or store an attribute, skips non-existing attributes on load.
        auto optionalIO = [ar](const std::string& attribute, auto* location, size_t attrSize)
        {
            try
            {
                ar->stepAttribute("cooling::" + attribute, location, attrSize);
            }
            catch (std::out_of_range&)
            {
                if (ar->rank() == 0)
                {
                    std::cout << "Attribute cooling::" << attribute
                              << " not set in file or initializer, setting to default value " << *location << std::endl;
                }
            }
        };
        for (size_t i = 0; i < parameterNames.size(); i++)
        {
            std::visit([&](auto* location) { optionalIO(std::string(parameterNames[i]), location, 1); }, parameters[i]);
        }
        optionalIO("cooling::ct_crit", &ct_crit, 1);
    }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_ptr;
    using FieldVariant = std::variant<float*, double*, int*>;
    static std::vector<const char*> getParameterNames();
    std::vector<FieldVariant>       getParameters();
};
} // namespace cooling
