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
 * @brief Global and local energy reductions
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 * @author Aurelien Cavelan
 */

#pragma once

#include <vector>
#include <iostream>

#ifdef USE_MPI
#include "mpi.h"
#endif

namespace sph
{

template<typename T, class Dataset>
void localEnergyReduction(size_t startIndex, size_t endIndex, Dataset& d, T* eKin, T* eInt)
{
    const T* u  = d.u.data();
    const T* vx = d.vx.data();
    const T* vy = d.vy.data();
    const T* vz = d.vz.data();
    const T* m  = d.m.data();

    T eKinThread = 0.0, eIntThread = 0.0;
#pragma omp parallel for reduction(+ : eKinThread, eIntThread)
    for (size_t i = startIndex; i < endIndex; i++)
    {
        T vmod2 = vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i];

        eKinThread += 0.5 * m[i] * vmod2;
        eIntThread += u[i] * m[i];
    }

    *eKin = eKinThread;
    *eInt = eIntThread;
}

/*! @brief global reduction of energies
 *
 * @tparam        T            float or double
 * @tparam        Dataset
 * @param[in]     startIndex   first locally assigned particle index of buffers in @p d
 * @param[in]     endIndex     last locally assigned particle index of buffers in @p d
 * @param[inout]  d            particle data set
 */
template<class Dataset>
void computeTotalEnergy(size_t startIndex, size_t endIndex, Dataset& d)
{
    using T = typename Dataset::RealType;
    T energies[3], globalEnergies[3];
    localEnergyReduction(startIndex, endIndex, d, energies + 0, energies + 1);

    energies[2] = d.egrav;

    int rootRank = 0;
#ifdef USE_MPI
    MPI_Reduce(energies, globalEnergies, 3, MpiType<T>{}, MPI_SUM, rootRank, MPI_COMM_WORLD);
#endif

    d.ecin  = globalEnergies[0];
    d.eint  = globalEnergies[1];
    d.egrav = globalEnergies[2];
    d.etot  = globalEnergies[0] + globalEnergies[1] + globalEnergies[2];
}

} // namespace sph
