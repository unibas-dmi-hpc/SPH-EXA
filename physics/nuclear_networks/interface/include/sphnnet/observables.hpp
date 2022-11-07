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
 * @brief Observables for nuclear networks (mainly energy).
 *
 * @author Joseph Touzet <joseph.touzet@ens-paris-saclay.fr>
 */

#pragma once

#include <numeric>
#include <omp.h>

#include "cstone/fields/data_util.hpp"

#include "nnet_util/eigen.hpp"
#include "nnet_util/algorithm.hpp"

// TODO header for nuclear energy calculation goes here. Equivalent to conserved_gpu.h

#include "mpi/mpi_wrapper.hpp"

namespace sphnnet
{
/*! @brief function to compute the total nuclear energy
 *
 * @param n     nuclearDataType containing a list of magnitude (named Y, being a vector of array)
 * @param BE    binding energy vector used to compute nuclear energy
 * @param comm  mpi communicator
 *
 * Returns the total nuclear binding energy (negative).
 */
template<class Data, typename Float>
Float totalNuclearEnergy(Data const& n, const Float* BE, MPI_Comm comm)
{
    const size_t n_particles = n.Y[0].size();
    const int    dimension   = n.numSpecies;

    if constexpr (cstone::HaveGpu<typename Data::AcceleratorType>{})
    {
        // TODO
        return 0.0;
    }
    std::vector<Float> Y(dimension);

    double localEnergy = 0;
#pragma omp parallel for firstprivate(Y) schedule(static) reduction(+ : localEnergy)
    for (size_t i = 0; i < n_particles; ++i)
    {
        // copy to local vector
        for (int j = 0; j < dimension; ++j)
            Y[j] = n.Y[j][i];

        localEnergy += -eigen::dot(Y.data(), Y.data() + dimension, BE) * n.m[i];
    }

    double totalEnergy = localEnergy;
    MPI_Allreduce(MPI_IN_PLACE, &totalEnergy, 1, MPI_DOUBLE, MPI_SUM, comm);

    return totalEnergy;
}
} // namespace sphnnet