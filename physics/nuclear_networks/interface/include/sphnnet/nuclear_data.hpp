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

#include <vector>
#include <array>
#include <memory>
#include <variant>

#include "CUDA/nuclear_data_stubs.hpp"

#if defined(USE_CUDA)
#include "CUDA/nuclear_data_gpu.cuh"
#endif

#include "mpi/mpi_wrapper.hpp"

#include "cstone/tree/accel_switch.hpp"
#include "cstone/fields/field_states.hpp"
#include "cstone/fields/data_util.hpp"
#include "cstone/fields/enumerate.hpp"
#include "cstone/util/reallocate.hpp"

namespace sphexa::sphnnet
{

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

    //! nuclear species are named "Yn"
    inline static constexpr std::array nuclearSpecies = enumerateFieldNames<"Y", maxNumSpecies>();

    //! detached hydro fields with same size as nuclearSpecies
    inline static constexpr std::array detachedHydroFields{
        "dt", "c", "p", "cv", "u", "dpdT", "m", "temp", "rho", "rho_m1", "nuclear_node_id", "nuclear_particle_id",
    };

    //! attached fields with same size as spatially distributed fields from SPH
    inline static constexpr std::array attachedFields{"node_id", "particle_id"};
    //! all detached fields
    inline static constexpr auto detachedFields = concatenate(nuclearSpecies, detachedHydroFields);
    //! the complete list with all fields
    inline static constexpr auto fieldNames = concatenate(detachedFields, attachedFields);

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

    /*! @brief sets the fields to be output
     *
     * @param outFields  vector of the names of fields to be output (including abundances names "Yi" for the ith
     * species)
     */
    void setOutputFields(const std::vector<std::string>& outFields)
    {
        outputFieldNames   = outFields;
        outputFieldIndices = cstone::fieldStringsToInt(outFields, fieldNames);
    }

    /*! @brief resize the number of particles of detached fields
     *
     * @param size  number of particles to be held by the class
     */
    void resize(size_t size)
    {
        devData.resize(size);

        double growthRate = 1;
        resizeFields(detachedFields, size, growthRate);
    }

    /*! @brief resize the number of particles of attached fields
     *
     * @param size  number of particles to be held by the class
     */
    void resizeAttached(size_t size)
    {
        devData.resizeAttached(size);

        double growthRate = 1.01;
        resizeFields(attachedFields, size, growthRate);
    }

    //! @brief particle fields selected for file output
    std::vector<int>         outputFieldIndices;
    std::vector<std::string> outputFieldNames;

private:
    template<class FieldList>
    void resizeFields(const FieldList& fields, size_t size, double growthRate)
    {
        auto data_ = data();
        for (auto field : fields)
        {
            size_t i = cstone::getFieldIndex(field, fieldNames);
            if (this->isAllocated(i))
            {
                std::visit([size, growthRate](auto& arg) { reallocate(*arg, size, growthRate); }, data_[i]);
            }
        }
    }

    template<size_t... Is>
    auto dataTuple_helper(std::index_sequence<Is...>)
    {
        return std::tie(Y[Is]...);
    }
};

} // namespace sphexa::sphnnet