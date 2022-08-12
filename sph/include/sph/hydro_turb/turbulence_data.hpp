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
 * @brief Contains the object holding all stirring/turbulence related data
 *
 * @author Axel Sanz <axelsanzlechuga@gmail.com>
 */

#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "cstone/cuda/cuda_utils.hpp"
#include "cstone/tree/accel_switch.hpp"

namespace sph
{

template<class T, class Accelerator>
class TurbulenceData
{
    // resulting type is a thrust::device_vector if the Accelerator is a GPU, otherwise a std::vector
    using DeviceVector =
        typename cstone::AccelSwitchType<Accelerator, std::vector, thrust::device_vector>::template type<T>;

public:
    using RealType = T;

    void resize(size_t numModes_)
    {
        modes.resize(numDim * numModes_);
        phases.resize(2 * numDim * numModes_);
        amplitudes.resize(numModes_);

        phasesReal.resize(numDim * numModes_);
        phasesImag.resize(numDim * numModes_);
    }

    void uploadModes()
    {
        if constexpr (cstone::HaveGpu<Accelerator>{})
        {
            // upload data to the GPU
            d_modes      = modes;
            d_amplitudes = amplitudes;
        }
    }

    const size_t numDim{3}; // Number of dimensions
    T            variance;  // Variance of Ornstein-Uhlenbeck process
    T            decayTime;
    T            solWeight; // Normalized Solenoidal weight

    std::mt19937 gen; // random engine

    size_t         numModes;   // Number of computed nodes
    std::vector<T> modes;      // Stirring Modes
    std::vector<T> phases;     // O-U Phases
    std::vector<T> amplitudes; // Amplitude of the modes

    std::vector<T> phasesReal; // created from phases on each step
    std::vector<T> phasesImag; // created from phases on each step

    DeviceVector d_modes;
    DeviceVector d_amplitudes;
    DeviceVector d_phasesReal;
    DeviceVector d_phasesImag;
};

} // namespace sph
