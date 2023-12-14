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
#include <limits>
#include <variant>
#include <vector>
#include <iostream>

#include "cstone/util/type_list.hpp"
#include "cstone/util/value_list.hpp"
#include "cooler_task.hpp"

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



    // Call the Grackle function f in parallel by dividing the data in bins of size N
    /* template<size_t N = 100, typename Tvec, typename F, typename... PType>
     void apply_N(const Tvec& rho, const Tvec& u, const ParticleType& particle, const size_t first, const size_t last,
                  F&& f, PType*... ptr)
     {
         const size_t n_particles = last - first;
         const size_t mod         = n_particles % N;
         const size_t ceil_term   = (mod == 0) ? 0 : 1;
         const size_t n_bins      = n_particles / N + ceil_term;
         const size_t N_last      = n_particles % n_bins;

         auto movePtr = [](const ParticleType& particle, const size_t diff)
         {
             auto f = [&](auto*... args) { return std::make_tuple(args + diff...); };
             return std::apply(f, particle);
         };

 #pragma omp parallel for
         for (size_t i = 0; i < n_bins; i++)
         {
             const size_t          len     = (i == n_bins - 1) ? N_last : N;
             const size_t          first_i = first + N * i;
             const size_t          last_i  = first_i + len;
             std::array<double, N> rho_copy{rho.begin() + first_i, rho.begin() + last_i};
             std::array<double, N> u_copy{u.begin() + first_i, u.begin() + last_i};
             std::array<double, N> return_qty;

             f(rho_copy.data(), u_copy.data(), movePtr(particle, first_i), len, return_qty.data());
         }
     }*/

    //! @brief Calls the GRACKLE library to integrate the cooling and chemistry fields
    void cool_particle(T dt, T& rho, T& u, const ParticleType& particle);
    void cool_particle_arr(T dt, T* rho, T* u, const ParticleType& particle, const size_t len);

    template<typename Tvec, typename Tvec2>
    void cool_particles(T dt, const Tvec& rho, const Tvec2& u, const ParticleType& particle, Tvec2 &du, const size_t first,
                        const size_t last)
    {
        constexpr size_t N = 1000;
        const task<N>    t(first, last);
        //printf("task: first: %d n_bins: %zu N_last: %zu n_part: %zu\n",
     //          t.first, t.n_bins, t.N_last, last - first);
        const auto       compute_du =
            [&](const auto& u_block, auto& du, const block& b)
        {
            for (size_t i = 0; i < b.len; i++)
            {
                du[i] = (u_block[i] - u[i + b.first]) / dt;
            }
        };

        //std::vector<double> rho_copy(rho.begin(), rho.end());
        //std::vector<double> u_copy(u.begin(), u.end());
        //std::vector<double> du_copy(u_copy.size());
#pragma omp parallel for
        for (size_t i = 0; i < t.n_bins; i++)
        {
            std::array<double, N> rho_copy;
            std::array<double, N> u_copy;
            std::array<double, N> du_local;
            const block           b(i, t);
            copyToBlock(rho, rho_copy, b);
            copyToBlock(u, u_copy, b);
            //printf("copied\n");

            auto particle_block = getBlockPointers(particle, b);
            //printf("copied\n");

            cool_particle_arr(dt, rho_copy.data(), u_copy.data(), particle_block, b.len);
            //cool_particle_arr(dt, rho_copy.data() + first, u_copy.data() + first, particle_block, b.len);

           // printf("copied\n");

            compute_du(u_copy, du_local, b);
            //compute_du(u_copy, du, b);
            //printf("copied\n");

            copyFromBlock(du_local, du, b);
            //printf("copied\n");

        }
    }

    //! @brief Calculate the temperature in K (physical units) from the internal energy (code units) and the chemistry
    //! composition
    T energy_to_temperature(T dt, T rho, T u, const ParticleType& particle);

    //! @brief Calculate pressure using the chemistry composition
    T    pressure(T rho, T u, const ParticleType& particle);
    void pressure_arr(T* rho, T* u, const ParticleType& particle, T* p, const size_t len);

    //! @brief Calculate adiabatic index from chemistry composition
    T    adiabatic_index(T rho, T u, const ParticleType& particle);
    void adiabatic_index_arr(T* rho, T* u, const ParticleType& particle, T* gamma, const size_t len);

    T    cooling_time(T rho, T u, const ParticleType& particle);
    void cooling_time_arr(T* rho, T* u, const ParticleType& particle, T* ct, const size_t len);

    template<typename Tvec, typename Tvec2>
    double min_cooling_time(const Tvec& rho, const Tvec2& u, const ParticleType& particle, const size_t first,
                                      const size_t last)
    {
        constexpr size_t    N = 1000;
        const task<N>       t(first, last);
        double ct_min = std::numeric_limits<double>::infinity();
#pragma omp parallel for reduction(min: ct_min)
        for (size_t i = 0; i < t.n_bins; i++)
        {
            const block           b(i, t);
            std::array<double, N> rho_copy;
            std::array<double, N> u_copy;
            std::array<double, N> ct_ret;

            copyToBlock(rho, rho_copy, b);
            copyToBlock(u, u_copy, b);

            auto particle_block = getBlockPointers(particle, b);
            cooling_time_arr(rho_copy.data(), u_copy.data(), particle_block, ct_ret.data(), b.len);
            std::transform(ct_ret.begin(), ct_ret.end(), ct_ret.begin(), [](double a) {return std::abs(a);});
            const double ct = *std::min_element(ct_ret.begin(), ct_ret.end());
            ct_min = std::min(ct_min, ct);
        }
        return ct_min * ct_crit;
    }
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
