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
 * @brief Utility functions to resolve names of particle fields to tuples of references
 *
 * Needs C++20 structural types
 */

#pragma once

#include "cstone/fields/data_util.hpp"
#include "cstone/tree/accel_switch.hpp"
#include "cstone/util/traits.hpp"
#include "cstone/util/util.hpp"

namespace cstone
{

template<StructuralString... Fields>
struct FieldList
{
};

template<StructuralString... Fields, class Dataset>
decltype(auto) getHost(Dataset& d)
{
    if constexpr (sizeof...(Fields) == 1)
    {
        return std::get<getFieldIndex(Fields.value..., Dataset::fieldNames)>(d.dataTuple());
    }
    else { return std::tie(std::get<getFieldIndex(Fields.value, Dataset::fieldNames)>(d.dataTuple())...); }
}

template<StructuralString... Fields, class Dataset>
decltype(auto) getHostHelper(Dataset& d, FieldList<Fields...>)
{
    return getHost<Fields...>(d);
}

template<class FieldList, class Dataset>
decltype(auto) getHost(Dataset& d)
{
    return getHostHelper(d, FieldList{});
}

template<StructuralString... Fields, class Dataset>
decltype(auto) getDevice(Dataset& d)
{
    if constexpr (sizeof...(Fields) == 1)
    {
        return std::get<getFieldIndex(Fields.value..., Dataset::fieldNames)>(d.devData.dataTuple());
    }
    else { return std::tie(std::get<getFieldIndex(Fields.value, Dataset::fieldNames)>(d.devData.dataTuple())...); }
}

template<StructuralString... Fields, class Dataset>
decltype(auto) getDeviceHelper(Dataset& d, FieldList<Fields...>)
{
    return getDevice<Fields...>(d);
}

template<class FieldList, class Dataset>
decltype(auto) getDevice(Dataset& d)
{
    return getDeviceHelper(d, FieldList{});
}

//! @brief Return a tuple of references to the specified particle field indices, to GPU fields if GPU is enabled
template<StructuralString... Fields, class Dataset>
decltype(auto) get(Dataset& d)
{
    using AcceleratorType = typename Dataset::AcceleratorType;
    if constexpr (std::is_same_v<AcceleratorType, CpuTag>) { return getHost<Fields...>(d); }
    else { return getDevice<Fields...>(d); }
}

template<StructuralString... Fields, class Dataset>
decltype(auto) getHelper(Dataset& d, FieldList<Fields...>)
{
    return get<Fields...>(d);
}

template<class FieldList, class Dataset>
decltype(auto) get(Dataset& d)
{
    return getHelper(d, FieldList{});
}

template<StructuralString... Fields>
constexpr auto make_tuple(FieldList<Fields...>)
{
    return std::make_tuple(Fields...);
}

} // namespace cstone
