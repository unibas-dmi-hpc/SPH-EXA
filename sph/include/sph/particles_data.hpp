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
 */

#pragma once

#include <array>
#include <cstdio>
#include <iostream>
#include <vector>

#include "sph/kernels.hpp"
#include "sph/tables.hpp"
#include "traits.hpp"

#if defined(USE_CUDA)
#include "sph/cuda/gpu_particle_data.cuh"
#endif

namespace sphexa
{

template<class Array>
std::vector<int> fieldStringsToInt(const Array&, const std::vector<std::string>&);

template<typename T, typename I, class AccType>
class ParticlesData
{
public:
    using RealType        = T;
    using KeyType         = I;
    using AcceleratorType = AccType;

    ParticlesData()
    {
        setConservedFields();
        setDependentFields();
    }

    size_t iteration{1};
    size_t n, side, count;

    T ttot{0.0}, etot{0.0}, ecin{0.0}, eint{0.0}, egrav{0.0};
    T minDt;

    std::vector<T> x, y, z, x_m1, y_m1, z_m1;    // Positions
    std::vector<T> vx, vy, vz;                   // Velocities
    std::vector<T> rho;                          // Density
    std::vector<T> u;                            // Internal Energy
    std::vector<T> p;                            // Pressure
    std::vector<T> h;                            // Smoothing Length
    std::vector<T> m;                            // Mass
    std::vector<T> c;                            // Speed of sound
    std::vector<T> grad_P_x, grad_P_y, grad_P_z; // gradient of the pressure
    std::vector<T> du, du_m1;                    // variation of the energy
    std::vector<T> dt, dt_m1;
    std::vector<T> c11, c12, c13, c22, c23, c33; // IAD components
    std::vector<T> maxvsignal;
    std::vector<T> mue, mui, temp, cv;
    std::vector<T> rho0, wrho0, kx, whomega, divv, curlv, alpha;

    std::vector<T> HI_fraction;
    std::vector<T> HII_fraction;
    std::vector<T> HM_fraction;
    std::vector<T> HeI_fraction;
    std::vector<T> HeII_fraction;
    std::vector<T> HeIII_fraction;
    std::vector<T> H2I_fraction;
    std::vector<T> H2II_fraction;
    std::vector<T> DI_fraction;
    std::vector<T> DII_fraction;
    std::vector<T> HDI_fraction;
    std::vector<T> e_fraction;
    std::vector<T> metal_fraction; //option: metal_cooling

    std::vector<T> volumetric_heating_rate; //option: use_volumetric_heating_rate
    std::vector<T> specific_heating_rate; //option: use_specific_heating_rate
    std::vector<T> RT_heating_rate; //option: use_radiative_transfer
    std::vector<T> RT_HI_ionization_rate; //option: use_radiative_transfer
    std::vector<T> RT_HeI_ionization_rate; //option: use_radiative_transfer
    std::vector<T> RT_HeII_ionization_rate; //option: use_radiative_transfer
    std::vector<T> RT_H2_dissociation_rate; //option: use_radiative_transfer
    std::vector<T> H2_self_shielding_length; //option: H2_self_shielding = 2

        std::vector<KeyType>                          codes;          // Particle space-filling-curve keys
    std::vector<int, PinnedAlloc_t<AccType, int>> neighborsCount; // number of neighbors of each particle
    std::vector<int>                              neighbors;      // only used in the CPU version

    DeviceData_t<AccType, T, KeyType> devPtrs;

    const std::array<double, lt::size> wh  = lt::createWharmonicLookupTable<double, lt::size>();
    const std::array<double, lt::size> whd = lt::createWharmonicDerivativeLookupTable<double, lt::size>();

    /*! @brief
     * Name of each field as string for use e.g in HDF5 output. Order has to correspond to what's returned by data().
     */
    inline static constexpr std::array fieldNames{"x",
                                                  "y",
                                                  "z",
                                                  "x_m1",
                                                  "y_m1",
                                                  "z_m1",
                                                  "vx",
                                                  "vy",
                                                  "vz",
                                                  "rho",
                                                  "u",
                                                  "p",
                                                  "h",
                                                  "m",
                                                  "c",
                                                  "grad_P_x",
                                                  "grad_P_y",
                                                  "grad_P_z",
                                                  "du",
                                                  "du_m1",
                                                  "dt",
                                                  "dt_m1",
                                                  "c11",
                                                  "c12",
                                                  "c13",
                                                  "c22",
                                                  "c23",
                                                  "c33",
                                                  "maxvsignal",
                                                  "mue",
                                                  "mui",
                                                  "temp",
                                                  "cv",
                                                  "rho0",
                                                  "wrho0",
                                                  "kx",
                                                  "whomega",
                                                  "divv",
                                                  "curlv",
                                                  "alpha",
                                                  "HI_fraction",
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
                                                  "H2_self_shielding_length"
    };

    /*! @brief return a vector of pointers to field vectors
     *
     * We implement this by returning an rvalue to prevent having to store pointers and avoid
     * non-trivial copy/move constructors.
     */
    auto data()
    {
        std::array<std::vector<T>*, fieldNames.size()> ret{&x,
                                                           &y,
                                                           &z,
                                                           &x_m1,
                                                           &y_m1,
                                                           &z_m1,
                                                           &vx,
                                                           &vy,
                                                           &vz,
                                                           &rho,
                                                           &u,
                                                           &p,
                                                           &h,
                                                           &m,
                                                           &c,
                                                           &grad_P_x,
                                                           &grad_P_y,
                                                           &grad_P_z,
                                                           &du,
                                                           &du_m1,
                                                           &dt,
                                                           &dt_m1,
                                                           &c11,
                                                           &c12,
                                                           &c13,
                                                           &c22,
                                                           &c23,
                                                           &c33,
                                                           &maxvsignal,
                                                           &mue,
                                                           &mui,
                                                           &temp,
                                                           &cv,
                                                           &rho0,
                                                           &wrho0,
                                                           &kx,
                                                           &whomega,
                                                           &divv,
                                                           &curlv,
                                                           &alpha,
                                                           &HI_fraction,
                                                           &HII_fraction,
                                                           &HM_fraction,
                                                           &HeI_fraction,
                                                           &HeII_fraction,
                                                           &HeIII_fraction,
                                                           &H2I_fraction,
                                                           &H2II_fraction,
                                                           &DI_fraction,
                                                           &DII_fraction,
                                                           &HDI_fraction,
                                                           &e_fraction,
                                                           &metal_fraction,
                                                           &volumetric_heating_rate,
                                                           &specific_heating_rate,
                                                           &RT_heating_rate,
                                                           &RT_HI_ionization_rate,
                                                           &RT_HeI_ionization_rate,
                                                           &RT_HeII_ionization_rate,
                                                           &RT_H2_dissociation_rate,
                                                           &H2_self_shielding_length,
        };

        static_assert(ret.size() == fieldNames.size());

        return ret;
    }

    void setConservedFields()
    {
        std::vector<std::string> fields{
            "x", "y", "z", "h", "m", "u", "vx", "vy", "vz", "x_m1", "y_m1", "z_m1", "du_m1", "dt_m1"};
        conservedFields = fieldStringsToInt(fieldNames, fields);
    }

    void setDependentFields()
    {
        std::vector<std::string> fields{"rho",
                                        "p",
                                        "c",
                                        "grad_P_x",
                                        "grad_P_y",
                                        "grad_P_z",
                                        "du",
                                        "dt",
                                        "c11",
                                        "c12",
                                        "c13",
                                        "c22",
                                        "c23",
                                        "c33",
                                        "maxvsignal"};

        dependentFields = fieldStringsToInt(fieldNames, fields);
    }

    void setConservedFieldsVE()
    {
        std::vector<std::string> fields{
            "x", "y", "z", "h", "m", "u", "vx", "vy", "vz", "x_m1", "y_m1", "z_m1", "du_m1", "dt_m1", "alpha"};
        conservedFields = fieldStringsToInt(fieldNames, fields);
    }

    void setDependentFieldsVE()
    {
        std::vector<std::string> fields{"rho",        "p",    "c",     "grad_P_x", "grad_P_y", "grad_P_z", "du",
                                        "dt",         "c11",  "c12",   "c13",      "c22",      "c23",      "c33",
                                        "maxvsignal", "rho0", "wrho0", "kx",       "whomega",  "divv",     "curlv"};
        dependentFields = fieldStringsToInt(fieldNames, fields);
    }

    void setOutputFields(const std::vector<std::string>& outFields)
    {
        outputFields = fieldStringsToInt(fieldNames, outFields);
    }

    //! @brief particle fields to conserve between iterations, needed for checkpoints and domain exchange
    std::vector<int> conservedFields;
    //! @brief particle fields recomputed every step from conserved fields
    std::vector<int> dependentFields;
    //! @brief particle fields select for file outputj
    std::vector<int> outputFields;

#ifdef USE_MPI
    MPI_Comm comm;
#endif
    int rank  = 0;
    int nrank = 1;

    // TODO: unify this with computePosition/Acceleration:
    // from SPH we have acceleration = -grad_P, so computePosition adds a factor of -1 to the pressure gradients
    // instead, the pressure gradients should be renamed to acceleration and computeMomentumAndEnergy should directly
    // set this to -grad_P, such that we don't need to add the gravitational acceleration with a factor of -1 on top
    T g = -1.0; // for Evrard Collapse Gravity.
    // constexpr static T g = 6.6726e-8; // the REAL value of g. g is 1.0 for Evrard mainly

    constexpr static T sincIndex     = 6.0;
    constexpr static T Kcour         = 0.2;
    constexpr static T maxDtIncrease = 1.1;

    // Min. Atwood number in ramp function in momentum equation (crossed/uncrossed selection)
    // Complete uncrossed option (Atmin>=1.d50, Atmax it doesn't matter).
    // Complete crossed (Atmin and Atmax negative)
    constexpr static T Atmax = 0.1;
    constexpr static T Atmin = 0.2;
    constexpr static T ramp  = 1.0 / (Atmax - Atmin);

    // AV switches floor and ceiling
    constexpr static T alphamin       = 0.05;
    constexpr static T alphamax       = 1.0;
    constexpr static T decay_constant = 0.2;

    // Interpolation kernel normalization constant
    const static T K;
};

template<typename T, typename I, class Acc>
const T ParticlesData<T, I, Acc>::K = sphexa::compute_3d_k(sincIndex);

template<class Array>
std::vector<int> fieldStringsToInt(const Array& allNames, const std::vector<std::string>& subsetNames)
{
    std::vector<int> subsetIndices;
    subsetIndices.reserve(subsetNames.size());
    for (const auto& field : subsetNames)
    {
        auto it = std::find(allNames.begin(), allNames.end(), field);
        if (it == allNames.end()) { throw std::runtime_error("Field " + field + " does not exist\n"); }

        size_t fieldIndex = it - allNames.begin();
        subsetIndices.push_back(fieldIndex);
    }
    return subsetIndices;
}

//! @brief extract a vector of pointers to particle fields for file output
template<class Dataset>
auto getOutputArrays(Dataset& dataset)
{
    using T            = typename Dataset::RealType;
    auto fieldPointers = dataset.data();

    std::vector<const T*> outputFields(dataset.outputFields.size());
    std::transform(dataset.outputFields.begin(),
                   dataset.outputFields.end(),
                   outputFields.begin(),
                   [&fieldPointers](int i) { return fieldPointers[i]->data(); });

    return outputFields;
}

//! @brief resizes all particles fields of @p d listed in data() to the specified size
template<class Dataset>
void resize(Dataset& d, size_t size)
{
    double growthRate = 1.05;
    auto   data_      = d.data();

    for (int i : d.conservedFields)
    {
        reallocate(*data_[i], size, growthRate);
    }
    for (int i : d.dependentFields)
    {
        reallocate(*data_[i], size, growthRate);
    }

    reallocate(d.codes, size, growthRate);
    reallocate(d.neighborsCount, size, growthRate);

    d.devPtrs.resize(size);
}

//! resizes the neighbors list, only used in the CPU verison
template<class Dataset>
void resizeNeighbors(Dataset& d, size_t size)
{
    double growthRate = 1.05;
    //! If we have a GPU, neighbors are calculated on-the-fly, so we don't need space to store them
    reallocate(d.neighbors, HaveGpu<typename Dataset::AcceleratorType>{} ? 0 : size, growthRate);
}

} // namespace sphexa
