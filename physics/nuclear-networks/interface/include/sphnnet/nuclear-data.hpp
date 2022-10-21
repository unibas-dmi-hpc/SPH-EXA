/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
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
 * @brief Definition of the main data class for nuclear networks, similar to the particuleData class in SPH-EXA.
 *
 * @author Joseph Touzet <joseph.touzet@ens-paris-saclay.fr>
 */

#pragma once

#include "CUDA/cuda.inl"

#include <vector>
#include <array>
#include <memory>
#include <variant>

#include "CUDA/nuclear-data-stubs.hpp"

#if defined(USE_CUDA)
#include "CUDA/nuclear-data-gpu.cuh"
#endif

#include "mpi/mpi-wrapper.hpp"

#include "cstone/tree/accel_switch.hpp"

#include "cstone/fields/field_states.hpp"
#include "cstone/fields/data_util.hpp"
#include "cstone/fields/enumerate.hpp"
#include "cstone/fields/concatenate.hpp"

#include "cstone/util/reallocate.hpp"

namespace sphexa::sphnnet
{

// TODO: naming conventions: file names should haves underscores, not hyphens: nuclear_data.hpp

/*! @brief nuclear data class for nuclear network */
template<typename RealType_, typename KeyType_, class Tmass_, class AccType>
struct NuclearDataType : public cstone::FieldStates<NuclearDataType<RealType_, KeyType_, Tmass_, AccType>>
{
public:
    //! maximum number of nuclear species
    static constexpr int maxNumSpecies = 100;
    //! actual number of nuclear species
    int numSpecies = 0;

    template<class ValueType>
    using FieldVector = std::vector<ValueType, std::allocator<ValueType>>;

    using RealType        = RealType_;
    using KeyType         = KeyType_;
    using Tmass           = Tmass_;
    using AcceleratorType = AccType;

    DeviceNuclearData_t<AcceleratorType, RealType, KeyType, Tmass> devData;

    //! nuclear energy
    RealType enuclear{0.0};
    // TODO: looks like iteration, ttot, minDt are duplicates of the same members in HydroData. Can you use these
    //       instead of duplicating?
    size_t   iteration{0};
    size_t   numParticlesGlobal;
    RealType ttot{0.0};
    //! current and previous (global) time-steps
    RealType minDt, minDt_m1;

    //! @brief hydro data
    FieldVector<RealType>                             c;                   // speed of sound
    FieldVector<RealType>                             p;                   // pressure
    FieldVector<RealType>                             cv;                  // cv
    FieldVector<RealType>                             u;                   // internal energy
    FieldVector<RealType>                             dpdT;                // dP/dT
    FieldVector<RealType>                             rho;                 // density
    FieldVector<RealType>                             temp;                // temperature
    FieldVector<RealType>                             rho_m1;              // previous density
    FieldVector<Tmass>                                m;                   // mass
    FieldVector<RealType>                             dt;                  // timesteps
    util::array<FieldVector<RealType>, maxNumSpecies> Y;                   // vector of nuclear abundances
    FieldVector<int>                                  nuclear_node_id;     // node ids (for nuclear data)
    FieldVector<KeyType>                              nuclear_particle_id; // particle id (for nuclear data)
    //! @brief data
    FieldVector<int>     node_id;     // node ids (for hydro data)
    FieldVector<KeyType> particle_id; // particle id (for hydro data)

    //! mpi partition
    sphexa::mpi::mpi_partition<KeyType> partition;

    //! fieldNames (every nuclear species is named "Yn")
    inline static constexpr auto fieldNames =
        concat(enumerateFieldNames<"Y", maxNumSpecies>(), std::array<const char*, 14>{
                                                              "dt",
                                                              "c",
                                                              "p",
                                                              "cv",
                                                              "u",
                                                              "dpdT",
                                                              "m",
                                                              "temp",
                                                              "rho",
                                                              "rho_m1",
                                                              "nuclear_node_id",
                                                              "nuclear_particle_id",
                                                              "node_id",
                                                              "particle_id",
                                                          });
    //! hydro sized field names
    inline static constexpr std::array hydroFields = {
        "node_id",
        "particle_id",
    };

    /*! @brief return a tuple of field references
     *
     * Note: this needs to be in the same order as listed in fieldNames
     */
    auto dataTuple()
    {
        return std::tuple_cat(dataTuple_helper(std::make_index_sequence<maxNumSpecies>{}),
                              std::tie(dt, c, p, cv, u, dpdT, m, temp, rho, rho_m1, nuclear_node_id,
                                       nuclear_particle_id, node_id, particle_id));
    }

    /*! @brief return a vector of pointers to field vectors
     *
     * We implement this by returning an rvalue to prevent having to store pointers and avoid
     * non-trivial copy/move constructors.
     */
    auto data()
    {
        using FieldType = std::variant<FieldVector<float>*, FieldVector<double>*, FieldVector<unsigned>*,
                                       FieldVector<int>*, FieldVector<uint64_t>*>;

        return std::apply([](auto&... fields) { return std::array<FieldType, sizeof...(fields)>{&fields...}; },
                          dataTuple());
    }

    /*! @brief sets the field to be outputed
     *
     * @param outFields  vector of the names of fields to be outputed (including abundances names "Y(i)" for the ith
     * species)
     */
    void setOutputFields(const std::vector<std::string>& outFields)
    {
        outputFieldNames   = outFields;
        outputFieldIndices = cstone::fieldStringsToInt(outFields, fieldNames);
    }

    /*! @brief resize the number of particles (nuclear)
     *
     * @param size  number of particles to be held by the class
     */
    void resize(size_t size)
    {
        double growthRate = 1;
        auto   data_      = data();

        devData.resize(size);

        for (size_t i = 0; i < data_.size(); ++i)
        {
            if ((!is_hydro(i)) && this->isAllocated(i))
            {
                // actually resize
                std::visit(
                    [&](auto& arg)
                    {
                        size_t previous_size = arg->size();

                        // reallocate
                        reallocate(*arg, size, growthRate);

                        // fill rho or rho_m1
                        if ((void*)arg == (void*)(&rho) || (void*)arg == (void*)(&rho_m1))
                        {
                            std::fill(arg->begin() + previous_size, arg->end(), 0.);
                        }

                    // TODO: a) this macro should not be here b) wrong place to initialize fields
                    //       if you initialize this field in nuclear_sedov_init.hpp, you'll solve both problems
#ifdef USE_NUCLEAR_NETWORKS
                        // fill dt
                        if ((void*)arg == (void*)(&dt))
                        {
                            std::fill(dt.begin() + previous_size, dt.end(), nnet::constants::initial_dt);
                        }
#endif
                    },
                    data_[i]);
            }
        }
    }

    /*! @brief resize the number of particles (hydro)
     *
     * @param size  number of particles to be held by the class
     */
    void resize_hydro(size_t size)
    {
        double growthRate = 1.05;
        auto   data_      = data();

        devData.resize_hydro(size);

        for (size_t i = 0; i < data_.size(); ++i)
        {
            if (is_hydro(i) && this->isAllocated(i))
            {
                // actually resize
                std::visit(
                    [&](auto& arg)
                    {
                        // reallocate
                        reallocate(*arg, size, growthRate);
                    },
                    data_[i]);
            }
        }
    }

    // particle fields selected for file output
    std::vector<int>         outputFieldIndices;
    std::vector<std::string> outputFieldNames;

private:
    bool is_hydro(int i) { return cstone::getFieldIndex(fieldNames[i], hydroFields) != hydroFields.size(); }

    template<size_t... Is>
    auto dataTuple_helper(std::index_sequence<Is...>)
    {
        return std::tie(Y[Is]...);
    }
};

} // namespace sphexa::sphnnet