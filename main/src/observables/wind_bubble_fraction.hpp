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
 * @brief output and calculate the bubble surviving fraction for the wind bubble shock test
 *
 */

#include <array>
#include <mpi.h>

#include "conserved_quantities.hpp"
#include "iobservables.hpp"
#include "io/file_utils.hpp"
#include "gpu_reductions.h"

namespace sphexa
{
//!@brief counts the number of particles that still belong to the cloud on each rank
template<class Tt, class T, class Tm>
size_t localSurvivors(size_t first, size_t last, const Tt* temp, const T* kx, const T* xmass, const Tm* m,
                      double rhoBubble, double tempWind)
{
    size_t survivors = 0;

#pragma omp parallel for reduction(+ : survivors)
    for (size_t i = first; i < last; i++)
    {
        T rhoi = kx[i] / xmass[i] * m[i];

        if (rhoi >= 0.64 * rhoBubble && temp[i] <= 0.9 * tempWind) { survivors++; }
    }

    return survivors;
}

/*!
 *
 * @brief calculates the percent of surviving cloud mass
 *
 * @param[in] first       index of first locally owned particle in @a u,kx,xmass fields
 * @param[in] last        index of last locally owned particle in @a u,kx,xmass fields
 * @param[in] u           internal energy
 * @param[in] kx          VE normalization
 * @param[in] xmass       VE definition
 * @param[in] m           particles masses
 * @param[in] rhoBubble   initial density inside the cloud
 * @param[in] uWind       initial internal energy of the supersonic wind
 * @param[in] initialMass initial total mass of the cloud
 * @return                fraction of particles surviving in the bubble
 *
 */
template<class Dataset>
double calculateSurvivingFraction(size_t first, size_t last, double rhoBubble, double uWind, double initialMass,
                                  Dataset& d)
{
    size_t localSurvived;

    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        localSurvived = survivorsGpu(rawPtr(d.devData.temp), rawPtr(d.devData.kx), rawPtr(d.devData.xm),
                                     rawPtr(d.devData.m), rhoBubble, uWind, first, last);
    }
    else
    {
        localSurvived =
            localSurvivors(first, last, d.temp.data(), d.kx.data(), d.xm.data(), d.m.data(), rhoBubble, uWind);
    }

    int    rootRank = 0;
    size_t globalSurvivors;
    MPI_Reduce(&localSurvived, &globalSurvivors, 1, MpiType<size_t>{}, MPI_SUM, rootRank, MPI_COMM_WORLD);

    return globalSurvivors * d.m[0] / initialMass;
}

//! @brief Observables that includes times, energies and bubble surviving fraction
template<class Dataset>
class WindBubble : public IObservables<Dataset>
{
    std::ofstream& constantsFile;
    double         rhoBubble;
    double         uWind;
    double         initialMass;

public:
    WindBubble(std::ofstream& constPath, double rhoInt, double uExt, double bubbleMass)
        : constantsFile(constPath)
        , rhoBubble(rhoInt)
        , uWind(uExt)
        , initialMass(bubbleMass)
    {
    }

    using T = typename Dataset::RealType;

    void computeAndWrite(Dataset& simData, size_t firstIndex, size_t lastIndex, cstone::Box<T>& box)
    {
        auto& d = simData.hydro;
        computeConservedQuantities(firstIndex, lastIndex, d, simData.comm);

        if (d.kx.empty())
        {
            throw std::runtime_error(
                "kx was empty. Wind Shock surviving fraction is only supported with volume elements (--prop ve)\n");
        }

        T    tempWind       = uWind * sph::idealGasCv(d.muiConst, d.gamma);
        auto bubbleFraction = calculateSurvivingFraction(firstIndex, lastIndex, rhoBubble, tempWind, initialMass, d);
        int  rank;
        MPI_Comm_rank(simData.comm, &rank);

        if (rank == 0)
        {
            T tkh            = 0.0937;
            T normalizedTime = d.ttot / tkh;

            fileutils::writeColumns(constantsFile, ' ', d.iteration, d.ttot, d.minDt, d.etot, d.ecin, d.eint, d.egrav,
                                    d.linmom, d.angmom, bubbleFraction, normalizedTime);
        }
    }
};

} // namespace sphexa
