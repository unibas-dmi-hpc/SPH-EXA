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
 * @brief output nuclear energies each iteration (default).
 * @author Joseph Touzet <joseph.touzet@ens-paris-saclay.fr>
 */

#include "sphnnet/observables.hpp"

#include "nnet/parameterization/net87/net87.hpp"
#include "nnet/parameterization/net14/net14.hpp"

#include "conserved_quantities.hpp"
#include "iobservables.hpp"
#include "io/ifile_writer.hpp"

namespace sphexa
{

template<class Dataset>
class NuclearEnergy : public IObservables<Dataset>
{
    std::ofstream& constantsFile;
    using T = typename Dataset::RealType;

    bool useAttached;

public:
    NuclearEnergy(std::ofstream& constPath, bool attached)
        : constantsFile(constPath)
        , useAttached(attached)
    {
    }

    void computeAndWrite(Dataset& simData, size_t firstIndex, size_t lastIndex, cstone::Box<T>& box)
    {
        auto&  d                   = simData.hydro;
        auto&  n                   = simData.nuclearData;
        size_t n_nuclear_particles = n.Y[0].size();

        int rank;
        MPI_Comm_rank(simData.comm, &rank);

        computeConservedQuantities(firstIndex, lastIndex, d, simData.comm);

        double const* BE;
        if (n.numSpecies == 14) { BE = nnet::net14::BE.data(); }
        else if (n.numSpecies == 86) { BE = nnet::net86::BE.data(); }
        else if (n.numSpecies == 87) { BE = nnet::net87::BE.data(); }
        else
        {
            throw std::runtime_error("not able to initialize " + std::to_string(n.numSpecies) + " nuclear species !");
        }

        if (!useAttached) { n.enuclear = sphnnet::totalNuclearEnergy(0, n_nuclear_particles, n, BE, simData.comm); }
        else { n.enuclear = sphnnet::totalNuclearEnergy(firstIndex, lastIndex, n, BE, simData.comm); }
        d.etot += n.enuclear;

        if (rank == 0)
        {
            fileutils::writeColumns(constantsFile, ' ', d.iteration, d.ttot, d.minDt, d.etot, n.enuclear, d.ecin,
                                    d.eint, d.egrav, d.linmom, d.angmom);
        }
    }
};

} // namespace sphexa
