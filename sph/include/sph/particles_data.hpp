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
 * @brief Contains the object holding all particle data
 *
 */

#pragma once

#include <array>
#include <vector>
#include <variant>

#include "cstone/cuda/cuda_utils.hpp"
#include "cstone/fields/data_util.hpp"
#include "cstone/fields/field_states.hpp"
#include "cstone/tree/accel_switch.hpp"
#include "cstone/tree/definitions.h"
#include "cstone/tree/octree.hpp"
#include "cstone/util/reallocate.hpp"

#include "sph/kernels.hpp"
#include "sph/tables.hpp"

#include "particles_data_stubs.hpp"

#if defined(USE_CUDA)
#include "sph/util/pinned_allocator.cuh"
#include "particles_data_gpu.cuh"
#endif

namespace sphexa
{

namespace lt = ::sph::lt;

template<typename T, typename KeyType_, class AccType>
class ParticlesData : public cstone::FieldStates<ParticlesData<T, KeyType_, AccType>>
{
public:
    using AcceleratorType = AccType;

    using KeyType   = KeyType_;
    using RealType  = T;
    using HydroType = float;
    using XM1Type   = float;
    using Tmass     = float;

    template<class ValueType>
    using PinnedVec = std::vector<ValueType, PinnedAlloc_t<AcceleratorType, ValueType>>;

    template<class ValueType>
    using FieldVector = std::vector<ValueType, std::allocator<ValueType>>;

    using FieldVariant =
        std::variant<FieldVector<float>*, FieldVector<double>*, FieldVector<unsigned>*, FieldVector<uint64_t>*>;

    ParticlesData() { devData.uploadTables(wh, whd); }
    ParticlesData(const ParticlesData&) = delete;

    uint64_t iteration{1};
    uint64_t numParticlesGlobal{0};

    //! @brief default mean desired number of neighbors per particle, can be overriden per test case or input file
    unsigned ng0{100};

    //! @brief default maximum number of neighbors per particle before additional h-adjustment will be triggered
    unsigned ngmax{150};

    T ttot{0.0}, etot{0.0}, ecin{0.0}, eint{0.0}, egrav{0.0};
    T linmom{0.0}, angmom{0.0};

    //! current and previous (global) time-steps
    T minDt{1e-12}, minDt_m1{1e-12};

    //! temporary MPI rank local timesteps;
    T minDtCourant{INFINITY}, minDtRho{INFINITY};
    //! @brief Fraction of Courant condition for timestep
    T Kcour{0.2};
    //! @brief Fraction of 1/|divv| condition for timestep
    T Krho{0.06};

    //! @brief gravitational constant
    T g{0.0};
    //! @brief gravitational smoothing
    T eps{0.005};
    //! @brief acceleration based time-step control
    T etaAcc{0.2};

    //! @brief adiabatic index
    T gamma{5.0 / 3.0};

    //! @brief mean molecular weight of ions for models that use one value for all particles
    Tmass muiConst{10.0};

    //! @brief Unified interface to attribute initialization, reading and writing
    template<class Archive>
    void loadOrStoreAttributes(Archive* ar)
    {
        //! @brief load or store an attribute, skips non-existing attributes on load.
        auto optionalIO = [ar](const std::string& attribute, auto* location, size_t attrSize)
        {
            try
            {
                ar->stepAttribute(attribute, location, attrSize);
            }
            catch (std::out_of_range&)
            {
                if (ar->rank() == 0)
                {
                    std::cout << "Attribute " << attribute << " not set in file, setting to default value " << *location
                              << std::endl;
                }
            }
        };

        ar->stepAttribute("iteration", &iteration, 1);
        ar->stepAttribute("numParticlesGlobal", &numParticlesGlobal, 1);
        optionalIO("ng0", &ng0, 1);
        optionalIO("ngmax", &ngmax, 1);
        ar->stepAttribute("time", &ttot, 1);
        ar->stepAttribute("minDt", &minDt, 1);
        ar->stepAttribute("minDt_m1", &minDt_m1, 1);
        optionalIO("Kcour", &Kcour, 1);
        optionalIO("Krho", &Krho, 1);
        ar->stepAttribute("gravConstant", &g, 1);
        optionalIO("gamma", &gamma, 1);
        optionalIO("eps", &eps, 1);
        optionalIO("etaAcc", &etaAcc, 1);
        optionalIO("muiConst", &muiConst, 1);
    }

    //! @brief non-stateful variables for statistics
    uint64_t totalNeighbors{0};

    /*! @brief Particle fields
     *
     * The length of these arrays equals the local number of particles including halos
     * if the field is active and is zero if the field is inactive.
     */
    FieldVector<T>         x, y, z;                            // Positions
    FieldVector<XM1Type>   x_m1, y_m1, z_m1;                   // Difference between current and previous positions
    FieldVector<HydroType> vx, vy, vz;                         // Velocities
    FieldVector<HydroType> rho;                                // Density
    FieldVector<T>         temp;                               // Temperature
    FieldVector<T>         u;                                  // Internal Energy
    FieldVector<HydroType> p;                                  // Pressure
    FieldVector<HydroType> prho;                               // p / (kx * m^2 * gradh)
    FieldVector<HydroType> h;                                  // Smoothing Length
    FieldVector<Tmass>     m;                                  // Mass
    FieldVector<HydroType> c;                                  // Speed of sound
    FieldVector<HydroType> cv;                                 // Specific heat
    FieldVector<HydroType> mue, mui;                           // mean molecular weight (electrons, ions)
    FieldVector<HydroType> divv, curlv;                        // Div(velocity), Curl(velocity)
    FieldVector<HydroType> ax, ay, az;                         // acceleration
    FieldVector<T>         du;                                 // energy rate of change (du/dt)
    FieldVector<XM1Type>   du_m1;                              // previous energy rate of change (du/dt)
    FieldVector<HydroType> c11, c12, c13, c22, c23, c33;       // IAD components
    FieldVector<HydroType> alpha;                              // AV coeficient
    FieldVector<HydroType> xm;                                 // Volume element definition
    FieldVector<HydroType> kx;                                 // Volume element normalization
    FieldVector<HydroType> gradh;                              // grad(h) term
    FieldVector<KeyType>   keys;                               // Particle space-filling-curve keys
    FieldVector<unsigned>  nc;                                 // number of neighbors of each particle
    FieldVector<HydroType> dV11, dV12, dV13, dV22, dV23, dV33; // Velocity gradient components

    //! @brief Indices of neighbors for each particle, length is number of assigned particles * ngmax. CPU version only.
    std::vector<cstone::LocalIndex>             neighbors;
    cstone::OctreeProperties<RealType, KeyType> treeView;

    DeviceData_t<AccType, T, KeyType> devData;

    const std::array<HydroType, lt::size> wh  = lt::createWharmonicTable<HydroType, lt::size>(sincIndex);
    const std::array<HydroType, lt::size> whd = lt::createWharmonicDerivativeTable<HydroType, lt::size>(sincIndex);

    /*! @brief
     * Name of each field as string for use e.g in HDF5 output. Order has to correspond to what's returned by data().
     */
    inline static constexpr std::array fieldNames{
        "x",     "y",    "z",   "x_m1", "y_m1", "z_m1", "vx",   "vy",   "vz",    "rho",  "u",     "p",
        "prho",  "h",    "m",   "c",    "ax",   "ay",   "az",   "du",   "du_m1", "c11",  "c12",   "c13",
        "c22",   "c23",  "c33", "mue",  "mui",  "temp", "cv",   "xm",   "kx",    "divv", "curlv", "alpha",
        "gradh", "keys", "nc",  "dV11", "dV12", "dV13", "dV22", "dV23", "dV33"};

    //! @brief dataset prefix to be prepended to fieldNames for structured output
    static const inline std::string prefix{};

    static_assert(!cstone::HaveGpu<AcceleratorType>{} ||
                      fieldNames.size() == DeviceData_t<AccType, T, KeyType>::fieldNames.size(),
                  "ParticlesData on CPU and GPU must have the same fields");

    /*! @brief return a tuple of field references
     *
     * Note: this needs to be in the same order as listed in fieldNames
     */
    auto dataTuple()
    {
        auto ret = std::tie(x, y, z, x_m1, y_m1, z_m1, vx, vy, vz, rho, u, p, prho, h, m, c, ax, ay, az, du, du_m1, c11,
                            c12, c13, c22, c23, c33, mue, mui, temp, cv, xm, kx, divv, curlv, alpha, gradh, keys, nc,
                            dV11, dV12, dV13, dV22, dV23, dV33);

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
        return std::apply([](auto&... fields) { return std::array<FieldVariant, sizeof...(fields)>{&fields...}; },
                          dataTuple());
    }

    /*! @brief mark fields file output
     *
     * @param outFields  list of field names
     *
     * Selected fields that match existing names contained in @a fieldNames will be removed from the argument
     * @p field names.
     */
    void setOutputFields(std::vector<std::string>& outFields)
    {
        auto hasField = [](const std::string& field)
        { return cstone::getFieldIndex(field, fieldNames) < fieldNames.size(); };

        std::copy_if(outFields.begin(), outFields.end(), std::back_inserter(outputFieldNames), hasField);
        outputFieldIndices = cstone::fieldStringsToInt(outputFieldNames, fieldNames);
        std::for_each(outputFieldNames.begin(), outputFieldNames.end(), [](auto& f) { f = prefix + f; });

        outFields.erase(std::remove_if(outFields.begin(), outFields.end(), hasField), outFields.end());
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

        devData.resize(size);
    }

    //! @brief particle fields selected for file output
    std::vector<int>         outputFieldIndices;
    std::vector<std::string> outputFieldNames;

    constexpr static T sincIndex     = 6.0;
    constexpr static T maxDtIncrease = 1.1;

    // Min. Atwood number in ramp function in momentum equation (crossed/uncrossed selection)
    // Complete uncrossed option (Atmin>=1.d50, Atmax it doesn't matter).
    // Complete crossed (Atmin and Atmax negative)
    constexpr static HydroType Atmin = 0.1;
    constexpr static HydroType Atmax = 0.2;
    constexpr static HydroType ramp  = 1.0 / (Atmax - Atmin);

    // AV switches floor and ceiling
    constexpr static HydroType alphamin       = 0.05;
    constexpr static HydroType alphamax       = 1.0;
    constexpr static HydroType decay_constant = 0.2;

    // Interpolation kernel normalization constant
    const static T K;
};

template<typename T, typename I, class Acc>
const T ParticlesData<T, I, Acc>::K = ::sph::compute_3d_k(sincIndex);

//! @brief resizes the neighbors list, only used in the CPU version
template<class Dataset>
void resizeNeighbors(Dataset& d, size_t size)
{
    double growthRate = 1.05;
    //! If we have a GPU, neighbors are calculated on-the-fly, so we don't need space to store them
    reallocate(d.neighbors, cstone::HaveGpu<typename Dataset::AcceleratorType>{} ? 0 : size, growthRate);
}

template<class Dataset, std::enable_if_t<not cstone::HaveGpu<typename Dataset::AcceleratorType>{}, int> = 0>
void transferToDevice(Dataset&, size_t, size_t, const std::vector<std::string>&)
{
}

template<class Dataset, std::enable_if_t<not cstone::HaveGpu<typename Dataset::AcceleratorType>{}, int> = 0>
void transferToHost(Dataset&, size_t, size_t, const std::vector<std::string>&)
{
}

template<class Vector, class T, std::enable_if_t<not IsDeviceVector<Vector>{}, int> = 0>
void fill(Vector& v, size_t first, size_t last, T value)
{
    std::fill(v.data() + first, v.data() + last, value);
}

} // namespace sphexa
