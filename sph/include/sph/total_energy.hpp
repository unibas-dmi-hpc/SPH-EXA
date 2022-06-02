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

#ifdef USE_MPI
#include "mpi.h"
#endif

namespace sph
{

template<typename T, class Dataset>
void localEnergyReduction(size_t startIndex, size_t endIndex, Dataset& d, T* eKin, T* eInt,
  T* linmomx, T* linmomy, T* linmomz, T* angmomx, T* angmomy, T* angmomz)
{
    const T* u  = d.u.data();
    const T* x  = d.x.data();
    const T* y  = d.y.data();
    const T* z  = d.z.data();
    const T* vx = d.vx.data();
    const T* vy = d.vy.data();
    const T* vz = d.vz.data();
    const T* m  = d.m.data();

    T eKinThread    = 0.0, eIntThread    = 0.0;
    T linmomxThread = 0.0, linmomyThread = 0.0, linmomzThread = 0.0;
    T angmomxThread = 0.0, angmomyThread = 0.0, angmomzThread = 0.0;

#pragma omp parallel for reduction(+ : eKinThread, eIntThread, linmomxThread, linmomyThread, linmomzThread, angmomxThread, angmomyThread, angmomzThread)
    for (size_t i = startIndex; i < endIndex; i++)
    {
        T vmod2 = vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i];

        eKinThread    += T(0.5) * m[i] * vmod2;
        eIntThread    += u[i] * m[i];
        linmomxThread += vx[i] * m[i];
        linmomyThread += vy[i] * m[i];
        linmomzThread += vz[i] * m[i];
        angmomxThread += (y[i] * vz[i] - z[i] * vy[i]) * m[i];
        angmomyThread += (z[i] * vx[i] - x[i] * vz[i]) * m[i];
        angmomzThread += (x[i] * vy[i] - y[i] * vx[i]) * m[i];
    }

    *eKin    = eKinThread;
    *eInt    = eIntThread;
    *linmomx = linmomxThread;
    *linmomy = linmomyThread;
    *linmomz = linmomzThread;
    *angmomx = angmomxThread;
    *angmomy = angmomyThread;
    *angmomz = angmomzThread;
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
    T energies[9], globalEnergies[9];
    localEnergyReduction(startIndex, endIndex, d, energies + 0, energies + 1,
      energies + 3, energies + 4, energies + 5, energies + 6, energies + 7, energies + 8);

    energies[2] = d.egrav;

    int rootRank = 0;
#ifdef USE_MPI
    MPI_Reduce(energies, globalEnergies, 9, MpiType<T>{}, MPI_SUM, rootRank, MPI_COMM_WORLD);
#endif

    d.ecin   = globalEnergies[0];
    d.eint   = globalEnergies[1];
    d.egrav  = globalEnergies[2];
    d.etot   = globalEnergies[0] + globalEnergies[1] + globalEnergies[2];
    d.linmom = std::sqrt(globalEnergies[3] * globalEnergies[3] + globalEnergies[4] * globalEnergies[4] + globalEnergies[5] * globalEnergies[5]);
    d.angmom = std::sqrt(globalEnergies[6] * globalEnergies[6] + globalEnergies[7] * globalEnergies[7] + globalEnergies[8] * globalEnergies[8]);
}

} // namespace sph
