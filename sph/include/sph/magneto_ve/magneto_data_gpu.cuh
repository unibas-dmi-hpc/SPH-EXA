/*
 * MIT License
 *
 * Copyright (c) 2024 CSCS, ETH Zurich, University of Basel, University of Zurich
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
 * @brief Contains the object holding hydrodynamical particle data on the GPU
 * @author Lukas Schmidt
 */

#pragma once

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <variant>

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/fields/field_states.hpp"
#include "cstone/primitives/primitives_gpu.h"
#include "cstone/tree/accel_switch.hpp"
#include "cstone/tree/definitions.h"
#include "cstone/util/reallocate.hpp"

namespace sphexa::magneto
{
class DeviceMagnetoData : public cstone::FieldStates<DeviceMagnetoData>
{

public:
    template<class FType>
    using DevVector = thrust::device_vector<FType>;

    using KeyType   = sph::SphTypes::KeyType;
    using RealType  = sph::SphTypes::CoordinateType;
    using HydroType = sph::SphTypes::HydroType;
    using XM1Type   = sph::SphTypes::XM1Type;
    using Tmass     = sph::SphTypes::Tmass;

    static constexpr int NST = 2;

    struct neighbors_stream
    {
        cudaStream_t stream;
    };

    struct neighbors_stream d_stream[NST];

    /*! @brief Particle fields
     *
     * The length of these arrays equals the local number of particles including halos
     * if the field is active and is zero if the field is inactive.
     */
    DevVector<RealType> Bx, By, Bz;             // Magnetic field components
    DevVector<RealType> dBx, dBy, dBz;          // Magnetic field rate of change (dB_i/dt)
    DevVector<XM1Type>  dBx_m1, dBy_m1, dBz_m1; // previous Magnetic field rate of change (dB_i/dt)

    // fields for divergence cleaning
    DevVector<HydroType> psi;      // scalar field used for divergence cleaning
    DevVector<HydroType> d_psi;    // rate of change of the scalar field (d psi/dt)
    DevVector<XM1Type>   d_psi_m1; // previous rate of change
    DevVector<HydroType> ch_m1;    // previous wave cleaning speed

    // Velocity Jacobian
    DevVector<HydroType> dvxdx, dvxdy, dvxdz;
    DevVector<HydroType> dvydx, dvydy, dvydz;
    DevVector<HydroType> dvzdx, dvzdy, dvzdz;

    // Magnetic field spatial derivatives
    DevVector<HydroType> divB;
    DevVector<HydroType> curlB_x, curlB_y, curlB_z;

    /*! @brief
     * Name of each field as string for use e.g in HDF5 output. Order has to correspond to what's returned by data().
     */
    inline static constexpr std::array fieldNames{
        "Bx",    "By",    "Bz",       "dBx",   "dBy",     "dBz",     "dBx_m1", "dBy_m1", "dBz_m1",
        "psi",   "d_psi", "d_psi_m1", "dvxdx", "dvxdy",   " dvxdz",  "dvydx",  "dvydy",  "dvydz",
        "dvzdx", "dvzdy", "dvzdz",    "divB",  "curlB_x", "curlB_y", "curlB_z", "ch_m1"};

    /*! @brief return a tuple of field references
     *
     * Note: this needs to be in the same order as listed in fieldNames
     */
    auto dataTuple()
    {
        auto ret = std::tie(Bx, By, Bz, dBx, dBy, dBz, dBx_m1, dBy_m1, dBz_m1, psi, d_psi, d_psi_m1, dvxdx, dvxdy,
                            dvxdz, dvydx, dvydy, dvydz, dvzdx, dvzdy, dvzdz, divB, curlB_x, curlB_y, curlB_z, ch_m1);

        static_assert(std::tuple_size_v<decltype(ret)> == fieldNames.size());
        return ret;
    }
    /*! @brief return a vector of pointers to field vectors
     *
     * We implement this by returning an rvalue to prevent having to store pointers and avoid
     * non-trivial copy/move constructors.
     */
    auto data()
    {
        using FieldType = std::variant<DevVector<float>*, DevVector<double>*, DevVector<unsigned>*,
                                       DevVector<uint64_t>*, DevVector<uint8_t>*>;

        return std::apply([](auto&... fields) { return std::array<FieldType, sizeof...(fields)>{&fields...}; },
                          dataTuple());
    }

    void resize(size_t size, float growthRate)
    {
        auto data_ = data();

        auto deallocateVector = [size](auto* devVectorPtr)
        {
            using DevVector = std::decay_t<decltype(*devVectorPtr)>;
            if (devVectorPtr->capacity() < size) { *devVectorPtr = DevVector{}; }
        };

        for (size_t i = 0; i < data_.size(); ++i)
        {
            if (this->isAllocated(i) && not this->isConserved(i)) { std::visit(deallocateVector, data_[i]); }
        }

        for (size_t i = 0; i < data_.size(); ++i)
        {
            if (this->isAllocated(i))
            {
                std::visit([size, growthRate](auto* arg) { reallocateDevice(*arg, size, growthRate); }, data_[i]);
            }
        }
    }

    size_t size()
    {
        auto data_ = data();
        for (size_t i = 0; i < data_.size(); ++i)
        {
            if (this->isAllocated(i))
            {
                return std::visit([](auto* arg) { return arg->size(); }, data_[i]);
            }
        }
        return 0;
    }

    DeviceMagnetoData()
    {
        for (int i = 0; i < NST; ++i)
        {
            checkGpuErrors(cudaStreamCreate(&d_stream[i].stream));
        }
    }

    ~DeviceMagnetoData()
    {
        for (int i = 0; i < NST; ++i)
        {
            checkGpuErrors(cudaStreamDestroy(d_stream[i].stream));
        }
    }
};

template<class DataType, std::enable_if_t<cstone::HaveGpu<typename DataType::AcceleratorType>{}, int> = 0>
void transferToDevice(DataType& d, size_t first, size_t last, const std::vector<std::string>& fields)
{
    auto hostData   = d.data();
    auto deviceData = d.devData.data();

    auto launchTransfer = [first, last](const auto* hostField, auto* deviceField)
    {
        using Type1 = std::decay_t<decltype(*hostField)>;
        using Type2 = std::decay_t<decltype(*deviceField)>;
        if constexpr (std::is_same_v<typename Type1::value_type, typename Type2::value_type>)
        {
            assert(hostField->size() > 0);
            assert(deviceField->size() > 0);
            size_t transferSize = (last - first) * sizeof(typename Type1::value_type);
            checkGpuErrors(cudaMemcpy(rawPtr(*deviceField) + first, hostField->data() + first, transferSize,
                                      cudaMemcpyHostToDevice));
        }
        else { throw std::runtime_error("Field type mismatch between CPU and GPU in copy to device"); }
    };

    for (const auto& field : fields)
    {
        int fieldIdx =
            std::find(DataType::fieldNames.begin(), DataType::fieldNames.end(), field) - DataType::fieldNames.begin();
        std::visit(launchTransfer, hostData[fieldIdx], deviceData[fieldIdx]);
    }
}

//! @brief transfer all specified fields allocated on both host and device to the device
template<class DataType, std::enable_if_t<cstone::HaveGpu<typename DataType::AcceleratorType>{}, int> = 0>
void transferAllocatedToDevice(DataType& d, size_t first, size_t last, const std::vector<std::string>& fields)
{
    for (const auto& field : fields)
    {
        if (d.isAllocated(field) && d.devData.isAllocated(field)) { transferToDevice(d, first, last, {field}); }
    }
}

template<class DataType, std::enable_if_t<cstone::HaveGpu<typename DataType::AcceleratorType>{}, int> = 0>
void transferToHost(DataType& d, size_t first, size_t last, const std::vector<std::string>& fields)
{
    auto hostData   = d.data();
    auto deviceData = d.devData.data();

    auto launchTransfer = [first, last](auto* hostField, const auto* deviceField)
    {
        using Type1 = std::decay_t<decltype(*hostField)>;
        using Type2 = std::decay_t<decltype(*deviceField)>;
        if constexpr (std::is_same_v<typename Type1::value_type, typename Type2::value_type>)
        {
            assert(hostField->size() > 0);
            assert(deviceField->size() > 0);
            size_t transferSize = (last - first) * sizeof(typename Type1::value_type);
            checkGpuErrors(cudaMemcpy(hostField->data() + first, rawPtr(*deviceField) + first, transferSize,
                                      cudaMemcpyDeviceToHost));
        }
        else { throw std::runtime_error("Field type mismatch between CPU and GPU in copy to device"); }
    };

    for (const auto& field : fields)
    {
        int fieldIdx =
            std::find(DataType::fieldNames.begin(), DataType::fieldNames.end(), field) - DataType::fieldNames.begin();
        std::visit(launchTransfer, hostData[fieldIdx], deviceData[fieldIdx]);
    }
}

template<class Vector, class T, std::enable_if_t<IsDeviceVector<Vector>{}, int> = 0>
void fill(Vector& v, size_t first, size_t last, T value)
{
    cstone::fillGpu(rawPtr(v) + first, rawPtr(v) + last, value);
}

} // namespace sphexa::magneto
