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
#include "io/ifile_writer.hpp"
#include "sph/math.hpp"

namespace sphexa
{
//!@brief counts the number of particles that still belong to the cloud on each rank
template<class T>
size_t localSurvivors(size_t first, size_t last, const T* u, const T* kx, const T* xmass, const T* m, double rhoBubble, double uWind)
{
    size_t survivors = 0;

#pragma omp parallel for reduction(+ : survivors)
    for (size_t i = first; i < last; i++)
    {
        T rhoi = kx[i] / xmass[i] * m[i];

        if (rhoi >= 0.64 * rhoBubble && u[i] <= 0.9 * uWind) survivors++;
    }

    return survivors;
}

/*!
 *
 * @brief calculates the percent of surviving cloud mass
 *
 * @param rhoBubble         initial density inside the cloud
 * @param uWind             initial internal energy of the supersonic wind
 * @param initialMass       initial total mass of the cloud
 *
 */
template<class T, class Dataset>
auto calculateSurvivingFraction(size_t startIndex, size_t endIndex, Dataset& d, double rhoBubble, double uWind,
                                double initialMass)
{

    size_t localSurvived = localSurvivors<T>(startIndex, endIndex, d.u.data(), d.kx.data(), d.xm.data(), d.m.data(), rhoBubble, uWind);

    int    rootRank = 0;
    size_t globalSurvivors;
    MPI_Reduce(&localSurvived, &globalSurvivors, 1, MpiType<size_t>{}, MPI_SUM, rootRank, MPI_COMM_WORLD);

    T globalFraction = globalSurvivors * d.m.data()[0] / initialMass;

    return globalFraction;
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

    void computeAndWrite(Dataset& d, size_t firstIndex, size_t lastIndex, cstone::Box<T>& box)
    {
        d.totalNeighbors = neighborsSum(firstIndex, lastIndex, d.neighborsCount);
        computeConservedQuantities(firstIndex, lastIndex, d);

        auto bubbleFraction =
            calculateSurvivingFraction<T, Dataset>(firstIndex, lastIndex, d, rhoBubble, uWind, initialMass);

        int rank;
        MPI_Comm_rank(d.comm, &rank);

        if (rank == 0)
        {
            T tkh            = 0.0937;
            T normalizedTime = d.ttot / tkh;

            fileutils::writeColumns(constantsFile,
                                    ' ',
                                    d.iteration,
                                    d.ttot,
                                    d.minDt,
                                    d.etot,
                                    d.ecin,
                                    d.eint,
                                    d.egrav,
                                    d.linmom,
                                    d.angmom,
                                    bubbleFraction,
                                    normalizedTime);
        }
    }
};

} // namespace sphexa
