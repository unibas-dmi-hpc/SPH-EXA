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
 * @brief Volume element definition i-loop driver
 *
 * @author Ruben Cabezon <ruben.cabezon@unibas.ch>
 */

#pragma once

#include "st_calcAccel.hpp"
#include "st_calcPhases.hpp"
#include "st_ounoise.hpp"
//#include "sph/sph.cuh"
//#include "sph/traits.hpp"

namespace sph
{
template<class Dataset>
void driver_turbulence2(size_t startIndex, size_t endIndex, Dataset& d)// const cstone::Box<T>& box)
{
  using T = typename Dataset::RealType;
  std::vector<T> st_aka(d.ndim*d.stNModes);
  std::vector<T> st_akb(d.ndim*d.stNModes);

  st_ounoiseupdate(d.stOUPhases, 6*d.stNModes, d.stOUvar, d.minDt, d.stDecay,d.stSeed);
  st_calcPhases(d.stNModes,d.ndim,d.stOUPhases,d.stSolWeight,d.stMode,st_aka,st_akb);
  st_calcAccel(startIndex,endIndex,d.ndim,d.x,d.y,d.z,d.ax,d.ay,d.az,d.stNModes,d.stMode,st_aka,st_akb,d.stAmpl,d.stSolWeightNorm);

}
/*
template<typename T, class Dataset>
void computeVeDefGradh(size_t startIndex, size_t endIndex, size_t ngmax, Dataset& d, const cstone::Box<T>& box)
{
    if constexpr (sphexa::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        cuda::computeVeDefGradh(startIndex, endIndex, ngmax, d, box);
    }
    else { computeVeDefGradhImpl(startIndex, endIndex, ngmax, d, box); }
}
*/
} // namespace sph
