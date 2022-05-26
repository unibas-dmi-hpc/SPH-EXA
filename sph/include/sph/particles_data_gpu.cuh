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
 * @brief Contains the object holding all particle data on the GPU
 */

#pragma once

#include <variant>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "cstone/util/util.hpp"
#include "util/cuda_utils.cuh"
#include "data_util.hpp"
#include "field_states.hpp"
#include "tables.hpp"
#include "traits.hpp"

namespace sphexa
{

template<typename T, class KeyType>
class DeviceParticlesData : public FieldStates<DeviceParticlesData<T, KeyType>>
{
    size_t allocatedTaskSize = 0;

    template<class FType>
    using DevVector = thrust::device_vector<FType>;

public:
    // number of CUDA streams to use
    static constexpr int NST = 2;
    // max number of particles to process per launch in kernel with async transfers
    static constexpr int taskSize = 1000000;

    struct neighbors_stream
    {
        cudaStream_t stream;
        int*         d_neighborsCount;
    };

    struct neighbors_stream d_stream[NST];

    /*! @brief Particle fields
     *
     * The length of these arrays equals the local number of particles including halos
     * if the field is active and is zero if the field is inactive.
     */
    DevVector<T>       x, y, z, x_m1, y_m1, z_m1;    // Positions
    DevVector<T>       vx, vy, vz;                   // Velocities
    DevVector<T>       rho;                          // Density
    DevVector<T>       temp;                         // Temperature
    DevVector<T>       u;                            // Internal Energy
    DevVector<T>       p;                            // Pressure
    DevVector<T>       prho;                         // p / (kx * m^2 * gradh)
    DevVector<T>       h;                            // Smoothing Length
    DevVector<T>       m;                            // Mass
    DevVector<T>       c;                            // Speed of sound
    DevVector<T>       cv;                           // Specific heat
    DevVector<T>       mue, mui;                     // mean molecular weight (electrons, ions)
    DevVector<T>       divv, curlv;                  // Div(velocity), Curl(velocity)
    DevVector<T>       ax, ay, az;                   // acceleration
    DevVector<T>       du, du_m1;                    // energy rate of change (du/dt)
    DevVector<T>       c11, c12, c13, c22, c23, c33; // IAD components
    DevVector<T>       alpha;                        // AV coeficient
    DevVector<T>       xm;                           // Volume element definition
    DevVector<T>       kx;                           // Volume element normalization
    DevVector<T>       gradh;                        // grad(h) term
    DevVector<KeyType> codes;                        // Particle space-filling-curve keys
    DevVector<int>     neighborsCount;               // number of neighbors of each particle

    DevVector<T> wh;
    DevVector<T> whd;

    /*! @brief
     * Name of each field as string for use e.g in HDF5 output. Order has to correspond to what's returned by data().
     */
    inline static constexpr std::array fieldNames{
        "x",   "y",   "z",   "x_m1", "y_m1", "z_m1", "vx", "vy",    "vz",    "rho",   "u",     "p",    "prho",
        "h",   "m",   "c",   "ax",   "ay",   "az",   "du", "du_m1", "c11",   "c12",   "c13",   "c22",  "c23",
        "c33", "mue", "mui", "temp", "cv",   "xm",   "kx", "divv",  "curlv", "alpha", "gradh", "keys", "nc"};

    /*! @brief return a vector of pointers to field vectors
     *
     * We implement this by returning an rvalue to prevent having to store pointers and avoid
     * non-trivial copy/move constructors.
     */
    auto data()
    {
        using IntVecType = std::decay_t<decltype(neighborsCount)>;
        using KeyVecType = std::decay_t<decltype(codes)>;
        using FieldType  = std::variant<DevVector<float>*, DevVector<double>*, KeyVecType*, IntVecType*>;

        std::array<FieldType, fieldNames.size()> ret{
            &x,   &y,   &z,   &x_m1, &y_m1, &z_m1, &vx, &vy,    &vz,    &rho,   &u,     &p,     &prho,
            &h,   &m,   &c,   &ax,   &ay,   &az,   &du, &du_m1, &c11,   &c12,   &c13,   &c22,   &c23,
            &c33, &mue, &mui, &temp, &cv,   &xm,   &kx, &divv,  &curlv, &alpha, &gradh, &codes, &neighborsCount};

        static_assert(ret.size() == fieldNames.size());

        return ret;
    }

    void resize(size_t size);

    DeviceParticlesData()
    {
        auto wh_table  = ::sph::lt::createWharmonicLookupTable<T, ::sph::lt::size>();
        auto whd_table = ::sph::lt::createWharmonicDerivativeLookupTable<T, ::sph::lt::size>();

        wh  = DevVector<T>(wh_table.begin(), wh_table.end());
        whd = DevVector<T>(whd_table.begin(), whd_table.end());

        for (int i = 0; i < NST; ++i)
        {
            CHECK_CUDA_ERR(cudaStreamCreate(&d_stream[i].stream));
        }
        resize_streams(taskSize);
    }

    ~DeviceParticlesData()
    {
        for (int i = 0; i < NST; ++i)
        {
            CHECK_CUDA_ERR(cudaStreamDestroy(d_stream[i].stream));
            CHECK_CUDA_ERR(::sph::cuda::utils::cudaFree(d_stream[i].d_neighborsCount));
        }
    }

private:
    void resize_streams(size_t newTaskSize)
    {
        if (newTaskSize > allocatedTaskSize)
        {
            if (allocatedTaskSize)
            {
                for (int i = 0; i < NST; ++i)
                {
                    CHECK_CUDA_ERR(::sph::cuda::utils::cudaFree(d_stream[i].d_neighborsCount));
                }
            }

            // allocate 1% extra to avoid reallocation on small size increase
            newTaskSize = size_t(double(newTaskSize) * 1.01);

            for (int i = 0; i < NST; ++i)
            {
                CHECK_CUDA_ERR(::sph::cuda::utils::cudaMalloc(newTaskSize * sizeof(int), d_stream[i].d_neighborsCount));
            }

            allocatedTaskSize = newTaskSize;
        }
    }
};

template<class ThrustVec>
typename ThrustVec::value_type* rawPtr(ThrustVec& p)
{
    assert(p.size() && "cannot get pointer to unallocated device vector memory");
    return thrust::raw_pointer_cast(p.data());
}

template<class ThrustVec>
const typename ThrustVec::value_type* rawPtr(const ThrustVec& p)
{
    assert(p.size() && "cannot get pointer to unallocated device vector memory");
    return thrust::raw_pointer_cast(p.data());
}

template<class DataType, std::enable_if_t<HaveGpu<typename DataType::AcceleratorType>{}, int> = 0>
void transferToDevice(DataType& d, size_t first, size_t last, const std::vector<std::string>& fields)
{
    auto hostData = d.data();
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
            CHECK_CUDA_ERR(cudaMemcpy(
                rawPtr(*deviceField) + first, hostField->data() + first, transferSize, cudaMemcpyHostToDevice));
        }
        else { throw std::runtime_error("Field type mismatch between CPU and GPU in copy to device");
        }
    };

    for (const auto& field : fields)
    {
        int fieldIdx =
            std::find(DataType::fieldNames.begin(), DataType::fieldNames.end(), field) - DataType::fieldNames.begin();
        std::visit(launchTransfer, hostData[fieldIdx], deviceData[fieldIdx]);
    }
}

template<class DataType, std::enable_if_t<HaveGpu<typename DataType::AcceleratorType>{}, int> = 0>
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
            CHECK_CUDA_ERR(cudaMemcpy(
                hostField->data() + first, rawPtr(*deviceField) + first, transferSize, cudaMemcpyDeviceToHost));
        }
        else { throw std::runtime_error("Field type mismatch between CPU and GPU in copy to device");
        }
    };

    for (const auto& field : fields)
    {
        int fieldIdx =
            std::find(DataType::fieldNames.begin(), DataType::fieldNames.end(), field) - DataType::fieldNames.begin();
        std::visit(launchTransfer, hostData[fieldIdx], deviceData[fieldIdx]);
    }
}

} // namespace sphexa
