/*
 * MIT License
 *
 * Copyright (c) 2022 CSCS, ETH Zurich, University of Zurich, University of Basel
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
 * @brief Unit tests for ParticlesData
 *
 * @author Noah Kubli <noah.kubli@uzh.ch>
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */
#pragma once

#include <array>
#include <optional>

#include "cstone/fields/enumerate.hpp"
#include "cstone/fields/field_states.hpp"
#include "cstone/fields/particles_get.hpp"
#include "cstone/util/reallocate.hpp"

namespace cooling
{

template<class T>
class ChemistryData : public cstone::FieldStates<ChemistryData<T>>
{
public:
    inline static constexpr size_t numFields = 21;

    template<class ValueType>
    using FieldVector     = std::vector<ValueType, std::allocator<ValueType>>;
    using RealType        = T;
    using AcceleratorType = cstone::CpuTag;

    std::array<FieldVector<T>, numFields> fields;

    auto dataTuple() { return dataTuple_helper(std::make_index_sequence<numFields>{}); }

    auto data()
    {
        using FieldType =
            std::variant<FieldVector<float>*, FieldVector<double>*, FieldVector<unsigned>*, FieldVector<uint64_t>*>;

        return std::apply([](auto&... fields) { return std::array<FieldType, sizeof...(fields)>{&fields...}; },
                          dataTuple());
    }

    void resize(size_t size)
    {
        double growthRate = 1.05;
        auto   data_      = data();

        for (size_t i = 0; i < data_.size(); ++i)
        {
            if (this->isAllocated(i))
            {
                std::visit([size, growthRate](auto& arg) { reallocate(*arg, size, growthRate); }, data_[i]);
            }
        }
    }

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

    static_assert(fieldNames.size() == numFields);

private:
    template<size_t... Is>
    auto dataTuple_helper(std::index_sequence<Is...>)
    {
        return std::tie(fields[Is]...);
    }
};

} // namespace cooling
