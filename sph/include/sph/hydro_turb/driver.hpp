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
 * @brief Turbulence driver calling all stirring subroutines
 *
 * @author Axel Sanz <axelsanzlechuga@gmail.com>
 */

#pragma once

#include "sph/hydro_turb/stirring.hpp"
#include "sph/hydro_turb/st_ounoise.hpp"
#include "sph/hydro_turb/phases.hpp"

#include "sph/util/cuda_stubs.h"
#ifdef USE_CUDA
#include "sph/util/cuda_utils.cuh"
#endif

namespace sph
{

/*! @brief Adds the stirring motion to particle accelerations
 *
 * @tparam Dataset
 * @param  startIndex
 * @param  endIndex
 * @param  d
 */
template<class Dataset>
void driveTurbulence(size_t startIndex, size_t endIndex, Dataset& d)
{
    using T = typename std::decay_t<decltype(d.turbulenceData)>::RealType;

    auto&          turb = d.turbulenceData;
    std::vector<T> phasesReal(turb.numDim * turb.numModes);
    std::vector<T> phasesImag(turb.numDim * turb.numModes);

    st_ounoiseupdate(turb.phases, turb.variance, d.minDt, turb.decayTime, turb.stSeed);
    computePhases(turb.numModes, turb.numDim, turb.phases, turb.stSolWeight, turb.modes, phasesReal, phasesImag);

    if constexpr (sphexa::HaveGpu<typename Dataset::AcceleratorType>{})
    {
#ifdef USE_CUDA
        thrust::device_vector<T> d_modes      = turb.modes;
        thrust::device_vector<T> d_phasesReal = phasesReal;
        thrust::device_vector<T> d_phasesImag = phasesImag;
        thrust::device_vector<T> d_amplitudes = turb.amplitudes;

        computeStirringGpu(startIndex,
                           endIndex,
                           turb.numDim,
                           rawPtr(d.devData.x),
                           rawPtr(d.devData.y),
                           rawPtr(d.devData.z),
                           rawPtr(d.devData.ax),
                           rawPtr(d.devData.ay),
                           rawPtr(d.devData.az),
                           turb.numModes,
                           rawPtr(d_modes),
                           rawPtr(d_phasesReal),
                           rawPtr(d_phasesImag),
                           rawPtr(d_amplitudes),
                           turb.solWeight);
#endif
    }
    else
    {
        computeStirring(startIndex,
                        endIndex,
                        turb.numDim,
                        d.x.data(),
                        d.y.data(),
                        d.z.data(),
                        d.ax.data(),
                        d.ay.data(),
                        d.az.data(),
                        turb.numModes,
                        turb.modes.data(),
                        phasesReal.data(),
                        phasesImag.data(),
                        turb.amplitudes.data(),
                        turb.solWeight);
    }
}

} // namespace sph
