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

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <optional>
#include <memory>
#include "cstone/util/util.hpp"
#include "cstone/fields/particles_get.hpp"


namespace cooling
{
    struct GlobalValues;
    struct chemistry_data_;

template <typename T>
struct Cooler
{
    Cooler();
    ~Cooler();

    void init(const double ms_sim, const double kp_sim, const int comoving_coordinates,
              const std::optional<chemistry_data_> grackleOptions     = std::nullopt,
              const std::optional<std::string>    grackleOptionsFile = std::nullopt,
              const std::optional<double>         t_sim              = std::nullopt);

    GlobalValues& get_global_values();
    chemistry_data_ getDefaultChemistryData();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_ptr;
};

//extern template struct Cooler<double>;
//extern template struct Cooler<float>;

//! @brief Initialize Grackle chemistry arrays with default data
    template<typename ChemistryData>
    void initGrackleData(ChemistryData& d, size_t n)
    {
    std::cout << "resizing: " << n << std::endl;
        using T = typename ChemistryData::RealType;
        // This is done so in the sample implementation from GRACKLE – don't know if really needed
        constexpr T tiny_number = 1.e-20;

        constexpr T metal_fraction     = 0.0;
        const T     non_metal_fraction = 1. - metal_fraction;
        //using gr_data                  = typename cooling::CoolingData<T>::FieldNames;

        d.resize(n);

        auto fillVec = [](std::vector<T>& vec, T value) { std::fill(vec.begin(), vec.end(), value); };

        fillVec(cstone::get<"HI_fraction">(d), non_metal_fraction * d.cooling_data.get_global_values().data.HydrogenFractionByMass);
        fillVec(cstone::get<"HeI_fraction">(d), non_metal_fraction * (1. - d.cooling_data.get_global_values().data.HydrogenFractionByMass));
        fillVec(cstone::get<"DI_fraction">(d), 2.0 * 3.4e-5);
        fillVec(cstone::get<"metal_fraction">(d), metal_fraction);
        fillVec(cstone::get<"HII_fraction">(d), tiny_number);
        fillVec(cstone::get<"HeII_fraction">(d), tiny_number);
        fillVec(cstone::get<"HeIII_fraction">(d), tiny_number);
        fillVec(cstone::get<"e_fraction">(d), tiny_number);
        fillVec(cstone::get<"HM_fraction">(d), tiny_number);
        fillVec(cstone::get<"H2I_fraction">(d), tiny_number);
        fillVec(cstone::get<"H2II_fraction">(d), tiny_number);
        fillVec(cstone::get<"DII_fraction">(d), tiny_number);
        fillVec(cstone::get<"HDI_fraction">(d), tiny_number);
        fillVec(cstone::get<"e_fraction">(d), tiny_number);
        fillVec(cstone::get<"metal_fraction">(d), tiny_number);
        fillVec(cstone::get<"volumetric_heating_rate">(d), tiny_number);
        fillVec(cstone::get<"specific_heating_rate">(d), tiny_number);
        fillVec(cstone::get<"RT_heating_rate">(d), tiny_number);
        fillVec(cstone::get<"RT_HI_ionization_rate">(d), tiny_number);
        fillVec(cstone::get<"RT_HeI_ionization_rate">(d), tiny_number);
        fillVec(cstone::get<"RT_HeII_ionization_rate">(d), tiny_number);
        fillVec(cstone::get<"RT_H2_dissociation_rate">(d), tiny_number);
        fillVec(cstone::get<"H2_self_shielding_length">(d), tiny_number);
        std::cout << "resizing: " << d.fields[0].size() << std::endl;

    }

//! @brief Calls the GRACKLE library to integrate the cooling and chemistry fields
        template<typename T>
        void cool_particle(GlobalValues& gv, const T& dt, T& rho, T& u, T& HI_fraction,
                           T& HII_fraction, T& HM_fraction, T& HeI_fraction, T& HeII_fraction, T& HeIII_fraction,
                           T& H2I_fraction, T& H2II_fraction, T& DI_fraction, T& DII_fraction, T& HDI_fraction, T& e_fraction,
                           T& metal_fraction, T& volumetric_heating_rate, T& specific_heating_rate, T& RT_heating_rate,
                           T& RT_HI_ionization_rate, T& RT_HeI_ionization_rate, T& RT_HeII_ionization_rate,
                           T& RT_H2_dissociation_rate, T& H2_self_shielding_length);

// Function not used now but may be needed for initializing the internal energy
        template<typename T>
        T energy_to_temperature(GlobalValues& gv, const T& dt, T& rho, T& u, T& HI_fraction,
                                T& HII_fraction, T& HM_fraction, T& HeI_fraction, T& HeII_fraction, T& HeIII_fraction,
                                T& H2I_fraction, T& H2II_fraction, T& DI_fraction, T& DII_fraction, T& HDI_fraction,
                                T& e_fraction, T& metal_fraction, T& volumetric_heating_rate, T& specific_heating_rate,
                                T& RT_heating_rate, T& RT_HI_ionization_rate, T& RT_HeI_ionization_rate,
                                T& RT_HeII_ionization_rate, T& RT_H2_dissociation_rate, T& H2_self_shielding_length);


    } // namespace cooling
