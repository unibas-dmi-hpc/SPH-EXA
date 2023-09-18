//
// Created by Noah Kubli on 29.11.22.
//

#pragma once

#include <iostream>
#include "cstone/fields/field_get.hpp"

namespace cooling
{
//! @brief Initialize Grackle chemistry arrays with default data
template<typename ChemistryData>
void initChemistryData(ChemistryData& d, size_t n)
{
    std::cout << "resizing: " << n << std::endl;
    using T = typename ChemistryData::RealType;
    // This is done so in the sample implementation from GRACKLE – don't know if really needed
    constexpr T tiny_number = 1.e-20;

    constexpr T metal_fraction     = 0.0;
    const T     non_metal_fraction = 1. - metal_fraction;

    d.resize(n);

    auto fillVec = [](std::vector<T>& vec, T value) { std::fill(vec.begin(), vec.end(), value); };

    fillVec(cstone::get<"HI_fraction">(d), non_metal_fraction * 0.76);
    fillVec(cstone::get<"HeI_fraction">(d), non_metal_fraction * 0.24);
    fillVec(cstone::get<"DI_fraction">(d), 2.0 * 3.4e-5);
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
    fillVec(cstone::get<"metal_fraction">(d), metal_fraction);
    fillVec(cstone::get<"volumetric_heating_rate">(d), 0.);
    fillVec(cstone::get<"specific_heating_rate">(d), 0.);
    fillVec(cstone::get<"RT_heating_rate">(d), 0.);
    fillVec(cstone::get<"RT_HI_ionization_rate">(d), 0.);
    fillVec(cstone::get<"RT_HeI_ionization_rate">(d), 0.);
    fillVec(cstone::get<"RT_HeII_ionization_rate">(d), 0.);
    fillVec(cstone::get<"RT_H2_dissociation_rate">(d), 0.);
    fillVec(cstone::get<"H2_self_shielding_length">(d), 0.);
    std::cout << "resizing: " << d.fields[0].size() << std::endl;
}
} // namespace cooling
