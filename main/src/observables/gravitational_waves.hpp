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

/*! @file output time, energies and the gravitational radiation as well as the quadrupole moment
 *
 *  @author Lukas Schmidt
 */

#include <array>
#include "mpi.h"
#include "iobservables.hpp"
#include "io/ifile_writer.hpp"
#include "grav_waves_calculations.hpp"

namespace sphexa
{

/*!@brief calculates the gravitational waves observable
 *
 * @tparam T            floating point type
 * @tparam Dataset
 * @param d             Dataset
 * @param first         first locally assigned particle index of buffers in @p d
 * @param last          first locally assigned particle index of buffers in @p d
 * @param viewTheta     viewing angle to determine ...
 * @param viewPhi       viewing angle to determine ...
 * @return              array containing the observables and the second derivative of the quadpole momentum:
 *                      {httplus, httcross, ixx, iyy, izz, ixy, ixz, iyz}
 */
template<class T, class Dataset>
std::array<T, 8> gravRad(Dataset& d,size_t first, size_t last, T viewTheta, T viewPhi)
{

    std::array<T, 6> d2Qxx_local;

    d2Qxx_local[0] = d2QuadpoleMomentum<T>(first, last, 0, 0,
                                       d.x.data(), d.y.data(), d.z.data(), d.vx.data(), d.vy.data(), d.vz.data(),
                                            d.ax.data(), d.ay.data(), d.az.data(), d.m.data()); //ixx
    d2Qxx_local[1] = d2QuadpoleMomentum<T>(first, last, 1, 1,
                                       d.x.data(), d.y.data(), d.z.data(), d.vx.data(), d.vy.data(), d.vz.data(),
                                            d.ax.data(), d.ay.data(), d.az.data(), d.m.data()); //iyy
    d2Qxx_local[2] = d2QuadpoleMomentum<T>(first, last, 2, 2,
                                       d.x.data(), d.y.data(), d.z.data(), d.vx.data(), d.vy.data(), d.vz.data(),
                                            d.ax.data(), d.ay.data(), d.az.data(), d.m.data()); //izz

    d2Qxx_local[3] = d2QuadpoleMomentum<T>(first, last, 0, 1,
                                       d.x.data(), d.y.data(), d.z.data(), d.vx.data(), d.vy.data(), d.vz.data(),
                                            d.ax.data(), d.ay.data(), d.az.data(), d.m.data()); //ixy
    d2Qxx_local[4] = d2QuadpoleMomentum<T>(first, last, 0, 2,
                                       d.x.data(), d.y.data(), d.z.data(), d.vx.data(), d.vy.data(), d.vz.data(),
                                            d.ax.data(), d.ay.data(), d.az.data(), d.m.data()); //ixz
    d2Qxx_local[5] = d2QuadpoleMomentum<T>(first, last, 1, 2,
                                       d.x.data(), d.y.data(), d.z.data(), d.vx.data(), d.vy.data(), d.vz.data(),
                                            d.ax.data(), d.ay.data(), d.az.data(), d.m.data()); //iyz


    int rootRank = 0;
    std::array<T, 6> d2Qxx_global;
    MPI_Reduce(d2Qxx_local.data(), d2Qxx_global.data(),6, MpiType<T>{}, MPI_SUM, rootRank, MPI_COMM_WORLD);

    T httplus;
    T httcross;

    computeHtt(d2Qxx_global, viewTheta, viewPhi, &httplus, &httcross);

    return {httplus, httcross, d2Qxx_global[0], d2Qxx_global[1], d2Qxx_global[2], d2Qxx_global[3], d2Qxx_global[4], d2Qxx_global[5]};
}


//! @brief Observables that includes times, energies, gravitational radiation and the second derivative of the quadrupole moment
template<class Dataset>
class GravWaves : public IObservables<Dataset>
{
    using T = typename Dataset::RealType;
    std::ofstream& constantsFile;
    T viewTheta;
    T viewPhi;

public:
    GravWaves(std::ofstream& constPath, T theta, T phi)
        : constantsFile(constPath), viewTheta(theta), viewPhi(phi)
    {
    }



    void computeAndWrite(Dataset& d, size_t firstIndex, size_t lastIndex, cstone::Box<T>& box)
    {

        std::array<T, 8> out = gravRad(d, firstIndex, lastIndex, viewTheta, viewPhi);

        int rank;
        MPI_Comm_rank(d.comm, &rank);

        if (rank == 0)
        {
            fileutils::writeColumns(
                constantsFile, ' ', d.iteration, d.ttot, d.minDt, d.etot, d.ecin, d.eint, d.egrav,
                out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]);
        }
    }
};
}