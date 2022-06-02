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
 * @brief Global and local energy, and linear and angular momentum reductions
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 * @author Aurelien Cavelan
 * @author Ruben Cabezon
 */

#pragma once

#include <vector>
#include <iostream>

#include "mpi.h"

#include "cstone/util/array.hpp"

namespace sphexa
{

template<class Dataset>
auto localConservedQuantities(size_t startIndex, size_t endIndex, Dataset& d)
{
    using T = typename Dataset::RealType;

    const T* x  = d.x.data();
    const T* y  = d.y.data();
    const T* z  = d.z.data();
    const T* vx = d.vx.data();
    const T* vy = d.vy.data();
    const T* vz = d.vz.data();
    const T* m  = d.m.data();
    const T* u  = d.u.data();

    T eKin = 0.0;
    T eInt = 0.0;

    util::array<T, 3> linmom{0.0, 0.0, 0.0};
    util::array<T, 3> angmom{0.0, 0.0, 0.0};

#pragma omp declare reduction(+ : util::array <T, 3> : omp_out += omp_in) initializer(omp_priv(omp_orig))

#pragma omp parallel for reduction(+ : eKin, eInt, linmom, angmom)
    for (size_t i = startIndex; i < endIndex; i++)
    {
        util::array<T, 3> X{x[i], y[i], z[i]};
        util::array<T, 3> V{vx[i], vy[i], vz[i]};
        auto              mi = m[i];

        eKin += mi * norm2(V);
        eInt += u[i] * mi;
        linmom += mi * V;
        angmom += mi * cross(X, V);
    }

    return std::make_tuple(T(0.5) * eKin, eInt, linmom, angmom);
}

/*! @brief Computation of globally conserved quantities
 *
 * @tparam        T            float or double
 * @tparam        Dataset
 * @param[in]     startIndex   first locally assigned particle index of buffers in @p d
 * @param[in]     endIndex     last locally assigned particle index of buffers in @p d
 * @param[inout]  d            particle data set
 */
template<class Dataset>
void computeConservedQuantities(size_t startIndex, size_t endIndex, Dataset& d)
{
    using T                           = typename Dataset::RealType;
    auto [eKin, eInt, linmom, angmom] = localConservedQuantities(startIndex, endIndex, d);

    T quantities[9], globalQuantities[9];

    quantities[0] = eKin;
    quantities[1] = eInt;
    quantities[2] = d.egrav;
    quantities[3] = linmom[0];
    quantities[4] = linmom[1];
    quantities[5] = linmom[2];
    quantities[6] = angmom[0];
    quantities[7] = angmom[1];
    quantities[8] = angmom[2];

    int rootRank = 0;
    MPI_Reduce(quantities, globalQuantities, 9, MpiType<T>{}, MPI_SUM, rootRank, MPI_COMM_WORLD);

    d.ecin  = globalQuantities[0];
    d.eint  = globalQuantities[1];
    d.egrav = globalQuantities[2];
    d.etot  = d.ecin + d.eint + d.egrav;

    util::array<T, 3> globalLinmom{globalQuantities[3], globalQuantities[4], globalQuantities[5]};
    util::array<T, 3> globalAngmom{globalQuantities[6], globalQuantities[7], globalQuantities[8]};
    d.linmom = std::sqrt(norm2(globalLinmom));
    d.angmom = std::sqrt(norm2(globalAngmom));
}

size_t neighborsSum(size_t startIndex, size_t endIndex, gsl::span<const int> neighborsCount)
{
    size_t sum = 0;
#pragma omp parallel for reduction(+ : sum)
    for (size_t i = startIndex; i < endIndex; i++)
    {
        sum += neighborsCount[i];
    }

    int    rootRank  = 0;
    size_t globalSum = 0;
    MPI_Reduce(&sum, &globalSum, 1, MpiType<size_t>{}, MPI_SUM, rootRank, MPI_COMM_WORLD);

    return globalSum;
}

} // namespace sphexa
