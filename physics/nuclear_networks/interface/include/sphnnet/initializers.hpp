
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
 * @brief Initializer functions for nuclear data.
 *
 * @author Joseph Touzet <joseph.touzet@ens-paris-saclay.fr>
 */

#pragma once

#include <numeric>

#include "mpi/mpi_wrapper.hpp"
#include "nuclear_net.hpp"

namespace sphnnet
{
/*! @brief intialize nuclear data, from a function of positions. Also initializes the partition correleating attached
 * and detached data.
 *
 * @param firstIndex   first (included) particle considered in d
 * @param lastIndex    last (excluded) particle considered in d
 * @param d            ParticlesDataType (contains positions)
 * @param n            nuclearDataType (to be populated)
 * @param initializer  function initializing nuclear abundances from position
 * @param comm         mpi communicator
 */
template<class NuclearDataType, class ParticlesDataType, class initFunc>
void initNuclearDataFromPos(size_t firstIndex, size_t lastIndex, ParticlesDataType& d, NuclearDataType& n,
                            const initFunc initializer, MPI_Comm comm)
{
    const int dimension = n.numSpecies;
    using Float         = typename std::remove_reference<decltype(n.Y[0][0])>::type;

    sphnnet::computePartition(firstIndex, lastIndex, n, comm);

    const size_t local_nuclear_n_particles = n.partition.recv_disp.back();

    // share the initial rho
    n.resize(local_nuclear_n_particles);

    // receiv position for initializer
    std::vector<Float> x(local_nuclear_n_particles), y(local_nuclear_n_particles), z(local_nuclear_n_particles);
    sphnnet::syncDataToStaticPartition(n.partition, d.x.data(), x.data(), d.comm);
    sphnnet::syncDataToStaticPartition(n.partition, d.y.data(), y.data(), d.comm);
    sphnnet::syncDataToStaticPartition(n.partition, d.z.data(), z.data(), d.comm);

    std::vector<Float> Y(dimension);

// intialize nuclear data
#pragma omp parallel for firstprivate(Y) schedule(dynamic)
    for (size_t i = 0; i < local_nuclear_n_particles; ++i)
    {
        Y = initializer(x[i], y[i], z[i]);
        for (int j = 0; j < dimension; ++j)
            n.Y[j][i] = Y[j];
    }
}

/*! @brief intialize nuclear data, from a function of radius. Also initializes the partition correleating attached and
 * detached data.
 *
 * @param firstIndex   first (included) particle considered in d
 * @param lastIndex    last (excluded) particle considered in d
 * @param d            ParticlesDataType (contains positions)
 * @param n            nuclearDataType (to be populated)
 * @param initializer  function initializing nuclear abundances from radius
 * @param comm         mpi communicator
 */
template<class NuclearDataType, class ParticlesDataType, class initFunc>
void initNuclearDataFromRadius(size_t firstIndex, size_t lastIndex, ParticlesDataType& d, NuclearDataType& n,
                               const initFunc initializer, MPI_Comm comm)
{
    const int dimension = n.numSpecies;
    using Float         = typename std::remove_reference<decltype(n.Y[0][0])>::type;

    sphnnet::computePartition(firstIndex, lastIndex, n, comm);

    const size_t local_nuclear_n_particles = n.partition.recv_disp.back();
    const size_t local_n_particles         = d.x.size();

    // receiv position for initializer
    std::vector<Float> send_r(local_n_particles), r(local_nuclear_n_particles, d.comm);

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < local_n_particles; ++i)
        send_r[i] = std::sqrt(d.x[i] * d.x[i] + d.y[i] * d.y[i] + d.z[i] * d.z[i]);

    sphnnet::syncDataToStaticPartition(n.partition, send_r.data(), r.data(), d.comm);

    std::vector<Float> Y(dimension);

// intialize nuclear data
#pragma omp parallel for firstprivate(Y) schedule(dynamic)
    for (size_t i = 0; i < local_nuclear_n_particles; ++i)
    {
        Y = initializer(r[i]);
        for (int j = 0; j < dimension; ++j)
            n.Y[j][i] = Y[j];
    }
}

/*! @brief intialize nuclear data, from a function of density. Also initializes the partition correleating attached and
 * detached data.
 *
 * @param firstIndex   first (included) particle considered in d
 * @param lastIndex    last (excluded) particle considered in d
 * @param d            ParticlesDataType (contains density)
 * @param n            nuclearDataType (to be populated)
 * @param initializer  function initializing nuclear abundances from radius
 * @param comm         mpi communicator
 */
template<class NuclearDataType, class ParticlesDataType, class initFunc>
void initNuclearDataFromRho(size_t firstIndex, size_t lastIndex, ParticlesDataType& d, NuclearDataType& n,
                            const initFunc initializer, MPI_Comm comm)
{
    const int dimension = n.numSpecies;
    using Float         = typename std::remove_reference<decltype(n.Y[0][0])>::type;

    sphnnet::computePartition(firstIndex, lastIndex, n, comm);

    const size_t local_nuclear_n_particles = n.partition.recv_disp.back();

    // share the initial rho
    n.resize(local_nuclear_n_particles);
    sphnnet::syncDataToStaticPartition(n.partition, d.rho.data(), n.rho.data(), d.comm);

    std::vector<Float> Y(dimension);

// intialize nuclear data
#pragma omp parallel for firstprivate(Y) schedule(dynamic)
    for (size_t i = 0; i < local_nuclear_n_particles; ++i)
    {
        Y = initializer(n.rho[i]);
        for (int j = 0; j < dimension; ++j)
            n.Y[j][i] = Y[j];
    }
}

/*! @brief intialize nuclear data, from a function of density. Also initializes the partition correleating attached and
 * detached data.
 *
 * @param firstIndex  first (included) particle considered in d
 * @param lastIndex   last (excluded) particle considered in d
 * @param d           ParticlesDataType (not used)
 * @param n           nuclearDataType (to be populated)
 * @param Y0          constant abundances vector to be copied
 * @param comm        mpi communicator
 */
template<class NuclearDataType, class ParticlesDataType, class Vector>
void initNuclearDataFromConst(size_t firstIndex, size_t lastIndex, ParticlesDataType& d, NuclearDataType& n,
                              const Vector& Y0, MPI_Comm comm)
{
    const int dimension = n.numSpecies;

    sphnnet::computePartition(firstIndex, lastIndex, n, comm);

    const size_t local_nuclear_n_particles = n.partition.recv_disp.back();

    // share the initial rho
    n.resize(local_nuclear_n_particles);

// intialize nuclear data
#pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < dimension; ++j)
        std::fill(n.Y[j].begin(), n.Y[j].end(), Y0[j]);
}

/*! @brief intialize nuclear data, from a function of positions. Also initializes the partition correleating attached
 * and detached data.
 *
 * @param firstIndex   first (included) particle considered in d
 * @param lastIndex    last (excluded) particle considered in d
 * @param d            ParticlesDataType, where d.nuclearData is the nuclear data container
 * @param initializer  function initializing nuclear abundances from position
 */
template<class initFunc, class ParticlesDataType>
void inline initNuclearDataFromPos(size_t firstIndex, size_t lastIndex, ParticlesDataType& d,
                                   const initFunc initializer)
{
    initNuclearDataFromPos(firstIndex, lastIndex, d.hydro, d.nuclearData, initializer, d.comm);
}

/*! @brief intialize nuclear data, from a function of radius. Also initializes the partition correleating attached and
 * detached data.
 *
 * @param firstIndex   first (included) particle considered in d
 * @param lastIndex    last (excluded) particle considered in d
 * @param d            ParticlesDataType, where d.nuclearData is the nuclear data container
 * @param initializer  function initializing nuclear abundances from radius
 */
template<class initFunc, class ParticlesDataType>
void inline initNuclearDataFromRadius(size_t firstIndex, size_t lastIndex, ParticlesDataType& d,
                                      const initFunc initializer)
{
    initNuclearDataFromRadius(firstIndex, lastIndex, d.hydro, d.nuclearData, initializer, d.comm);
}

/*! @brief intialize nuclear data, from a function of density. Also initializes the partition correleating attached and
 * detached data.
 *
 * @param firstIndex   first (included) particle considered in d
 * @param lastIndex    last (excluded) particle considered in d
 * @param d            ParticlesDataType, where d.nuclearData is the nuclear data container
 * @param initializer  function initializing nuclear abundances from radius
 */
template<class initFunc, class ParticlesDataType>
void inline initNuclearDataFromRho(size_t firstIndex, size_t lastIndex, ParticlesDataType& d,
                                   const initFunc initializer)
{
    initNuclearDataFromRadius(firstIndex, lastIndex, d.hydro, d.nuclearData, initializer, d.comm);
}

/*! @brief intialize nuclear data, from a function of density. Also initializes the partition correleating attached and
 * detached data.
 *
 * @param firstIndex   first (included) particle considered in d
 * @param lastIndex    last (excluded) particle considered in d
 * @param d            ParticlesDataType, where d.nuclearData is the nuclear data container
 * @param initializer  function initializing nuclear abundances from radius
 */
template<class Vector, class ParticlesDataType>
void inline initNuclearDataFromConst(size_t firstIndex, size_t lastIndex, ParticlesDataType& d, const Vector& Y0)
{
    initNuclearDataFromConst(firstIndex, lastIndex, d.hydro, d.nuclearData, Y0, d.comm);
}
} // namespace sphnnet