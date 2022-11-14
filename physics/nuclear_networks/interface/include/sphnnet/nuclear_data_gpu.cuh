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
 * @brief Definition of CUDA GPU data.
 *
 * @author Joseph Touzet <joseph.touzet@ens-paris-saclay.fr>
 */

#pragma once

#include <vector>
#include <array>
#include <memory>

#include <thrust/device_vector.h>

#include "cstone/fields/field_states.hpp"
#include "cstone/fields/enumerate.hpp"
#include "cstone/util/reallocate.hpp"
#include "cstone/util/array.hpp"

namespace sphnnet
{
/*! @brief device nuclear data class for nuclear network */
template<typename RealType_, typename KeyType_, typename Tmass_>
class DeviceNuclearDataType : public cstone::FieldStates<DeviceNuclearDataType<RealType_, KeyType_, Tmass_>>
{
public:
    //! maximum number of nuclear species
    static const int maxNumSpecies = 100;
    //! actual number of nuclear species
    int numSpecies = 0;

    template<class FType>
    using DevVector = thrust::device_vector<FType>;

    using RealType = RealType_;
    using KeyType  = KeyType_;
    using Tmass    = Tmass_;

    DevVector<RealType>                             c;                   // speed of sound
    DevVector<RealType>                             p;                   // pressure
    DevVector<RealType>                             cv;                  // cv
    DevVector<RealType>                             u;                   // internal energy
    DevVector<RealType>                             dpdT;                // dP/dT
    DevVector<RealType>                             rho;                 // density
    DevVector<RealType>                             temp;                // temperature
    DevVector<RealType>                             rho_m1;              // previous density
    DevVector<Tmass>                                m;                   // mass
    DevVector<RealType>                             dt;                  // timesteps
    util::array<DevVector<RealType>, maxNumSpecies> Y;                   // vector of nuclear abundances
    DevVector<int>                                  nuclear_node_id;     // node ids (for nuclear data)
    DevVector<KeyType>                              nuclear_particle_id; // particle id (for nuclear data)
    //! @brief data
    DevVector<int>     node_id;     // node ids (for hydro data)
    DevVector<KeyType> particle_id; // particle id (for hydro data)

    mutable thrust::device_vector<RealType> buffer; // solver buffer

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
        using FieldType = std::variant<DevVector<float>*, DevVector<double>*, DevVector<unsigned>*, DevVector<int>*,
                                       DevVector<uint64_t>*>;

        return std::apply([](auto&... fields) { return std::array<FieldType, sizeof...(fields)>{&fields...}; },
                          dataTuple());
    }

    //! @brief resize the number of particles of detached fields
    void resize(size_t size)
    {
        double growthRate = 1;
        resizeFields(detachedFields, size, growthRate);
    }

    //! @brief resize the number of particles of attached fields
    void resizeAttached(size_t size)
    {
        double growthRate = 1.01;
        resizeFields(attachedFields, size, growthRate);
    }

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

} // namespace sphnnet