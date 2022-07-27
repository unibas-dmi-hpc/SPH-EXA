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

#include "stirring.hpp"
#include "phases.hpp"
#include "st_ounoise.hpp"

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
    using T = typename Dataset::RealType;

    auto&          turb = d.turbulenceData;
    std::vector<T> st_aka(turb.numDim * turb.numModes);
    std::vector<T> st_akb(turb.numDim * turb.numModes);

    st_ounoiseupdate(turb.phases, turb.variance, d.minDt, turb.decayTime, turb.stSeed);
    st_calcPhases(turb.numModes, turb.numDim, turb.phases, turb.stSolWeight, turb.modes, st_aka, st_akb);
    computeStirring(startIndex,
                    endIndex,
                    turb.numDim,
                    d.x,
                    d.y,
                    d.z,
                    d.ax,
                    d.ay,
                    d.az,
                    turb.numModes,
                    turb.modes,
                    st_aka,
                    st_akb,
                    turb.amplitudes,
                    turb.solWeight);
}

} // namespace sph
