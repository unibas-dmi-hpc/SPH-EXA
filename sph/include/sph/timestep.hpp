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
 * @brief Min-reduction to determine global timestep
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 * @author Aurelien Cavelan
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <vector>
#include <mpi.h>

#include "kernels.hpp"

namespace sph
{

//! @brief limit time-step based on accelerations when gravity is enabled
template<class Dataset>
auto accelerationTimestep(size_t first, size_t last, const Dataset& d)
{
    using T = typename Dataset::RealType;

    T maxAccSq = 0.0;
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        maxAccSq = cstone::maxNormSquareGpu(rawPtr(d.devData.ax) + first, rawPtr(d.devData.ay) + first,
                                            rawPtr(d.devData.az) + first, last - first);
    }
    else
    {
#pragma omp parallel for reduction(max : maxAccSq)
        for (size_t i = first; i < last; ++i)
        {
            cstone::Vec3<T> X{d.ax[i], d.ay[i], d.az[i]};
            maxAccSq = std::max(norm2(X), maxAccSq);
        }
    }

    return d.etaAcc * std::sqrt(d.eps / std::sqrt(maxAccSq));
}

//! @brief limit time-step based on divergence of velocity
template<class Dataset>
auto rhoTimestep(size_t first, size_t last, const Dataset& d)
{
    using T = typename Dataset::RealType;

    T maxDivv = -INFINITY;
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        if (d.devData.divv.empty()) { throw std::runtime_error("Divv needs to be available in rhoTimestep\n"); }
        auto minmax = cstone::MinMaxGpu<T>{}(rawPtr(d.devData.divv) + first, rawPtr(d.devData.divv) + last);
        maxDivv     = std::get<1>(minmax);
    }
    else
    {
        if (d.divv.empty()) { throw std::runtime_error("Divv needs to be available in rhoTimestep\n"); }

#pragma omp parallel for reduction(max : maxDivv)
        for (size_t i = first; i < last; ++i)
        {
            maxDivv = std::max(d.divv[i], maxDivv);
        }
    }
    return d.Krho / std::abs(maxDivv);
}

template<class Dataset>
void computeTimestep(size_t first, size_t last, Dataset& d)
{
    using T = typename Dataset::RealType;

    T minDtAcc = (d.g != 0.0) ? accelerationTimestep(first, last, d) : INFINITY;

    T minDtLoc = std::min({minDtAcc, d.minDtCourant, d.minDtRho, d.maxDtIncrease * d.minDt});

    T minDtGlobal;
    MPI_Allreduce(&minDtLoc, &minDtGlobal, 1, MpiType<T>{}, MPI_MIN, MPI_COMM_WORLD);

    d.ttot += minDtGlobal;

    d.minDt_m1 = d.minDt;
    d.minDt    = minDtGlobal;
}
template<class Dataset, typename Cooler, typename Chem>
void computeTimestep_cool(size_t first, size_t last, Dataset& d, Cooler& cooler, Chem& chem)
{
    using T = typename Dataset::RealType;

    T minDtAcc = (d.g != 0.0) ? accelerationTimestep(first, last, d) : INFINITY;

    T minTc = minDtAcc;
//    T minTc(INFINITY);
//#pragma omp parallel for reduction(min : minTc)
//    for (size_t i = first; i < last; i++)
//    {
//        const T cooling_time = cooler.cooling_time(
//            d.rho[i], d.u[i], get<"HI_fraction">(chem)[i], get<"HII_fraction">(chem)[i], get<"HM_fraction">(chem)[i],
//            get<"HeI_fraction">(chem)[i], get<"HeII_fraction">(chem)[i], get<"HeIII_fraction">(chem)[i],
//            get<"H2I_fraction">(chem)[i], get<"H2II_fraction">(chem)[i], get<"DI_fraction">(chem)[i],
//            get<"DII_fraction">(chem)[i], get<"HDI_fraction">(chem)[i], get<"e_fraction">(chem)[i],
//            get<"metal_fraction">(chem)[i], get<"volumetric_heating_rate">(chem)[i],
//            get<"specific_heating_rate">(chem)[i], get<"RT_heating_rate">(chem)[i],
//            get<"RT_HI_ionization_rate">(chem)[i], get<"RT_HeI_ionization_rate">(chem)[i],
//            get<"RT_HeII_ionization_rate">(chem)[i], get<"RT_H2_dissociation_rate">(chem)[i],
//            get<"H2_self_shielding_length">(chem)[i]);
//        minTc = std::min(std::abs(cooling_time), minTc);
//    }
//    std::cout << "minTc: " << minTc << std::endl;
    T minDtLoc = std::min({minDtAcc, d.minDtCourant, d.minDtRho, d.maxDtIncrease * d.minDt, 0.01 * minTc});

    T minDtGlobal;
    MPI_Allreduce(&minDtLoc, &minDtGlobal, 1, MpiType<T>{}, MPI_MIN, MPI_COMM_WORLD);

    d.ttot += minDtGlobal;

    d.minDt_m1 = d.minDt;
    d.minDt    = minDtGlobal;
}

} // namespace sph
