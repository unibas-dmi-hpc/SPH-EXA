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

namespace cooling
{

template<typename T>
struct Cooler
{
    Cooler();

    ~Cooler();

    //! @brief Init Cooler. Must be called before any other function is used.
    void init(const double ms_sim, const double kp_sim, const int comoving_coordinates,
              const std::optional<std::map<std::string, std::any>> grackleOptions = std::nullopt,
              const std::optional<double>                          t_sim          = std::nullopt);

    //! @brief Calls the GRACKLE library to integrate the cooling and chemistry fields
    void cool_particle(const T& dt, T& rho, T& u, T& HI_fraction, T& HII_fraction, T& HM_fraction, T& HeI_fraction,
                       T& HeII_fraction, T& HeIII_fraction, T& H2I_fraction, T& H2II_fraction, T& DI_fraction,
                       T& DII_fraction, T& HDI_fraction, T& e_fraction, T& metal_fraction, T& volumetric_heating_rate,
                       T& specific_heating_rate, T& RT_heating_rate, T& RT_HI_ionization_rate,
                       T& RT_HeI_ionization_rate, T& RT_HeII_ionization_rate, T& RT_H2_dissociation_rate,
                       T& H2_self_shielding_length);

    //! @brief Function not used now but may be needed for initializing the internal energy
    T energy_to_temperature(const T& dt, T& rho, T& u, T& HI_fraction, T& HII_fraction, T& HM_fraction, T& HeI_fraction,
                            T& HeII_fraction, T& HeIII_fraction, T& H2I_fraction, T& H2II_fraction, T& DI_fraction,
                            T& DII_fraction, T& HDI_fraction, T& e_fraction, T& metal_fraction,
                            T& volumetric_heating_rate, T& specific_heating_rate, T& RT_heating_rate,
                            T& RT_HI_ionization_rate, T& RT_HeI_ionization_rate, T& RT_HeII_ionization_rate,
                            T& RT_H2_dissociation_rate, T& H2_self_shielding_length);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_ptr;
};
} // namespace cooling
