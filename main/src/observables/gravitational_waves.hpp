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
 * @param viewTheta     viewing angle for the polarization modes
 * @param viewPhi       viewing angle for the polarization modes
 * @return              array containing the polarization modes and the second derivative of the quadpole momentum:
 *                      {httplus, httcross, ixx, iyy, izz, ixy, ixz, iyz}
 */
template<class T, class Dataset>
auto gravRad(Dataset& d, size_t first, size_t last, T viewTheta, T viewPhi)
{

    struct Dim
    {
        enum IndexLabels
        {
            x = 0,
            y = 1,
            z = 2,
        };
    };

    std::array<T, 6> d2Q_local;

    d2Q_local[QIdx::xx] = d2QuadpoleMomentum<T>(first,
                                                last,
                                                Dim::x,
                                                Dim::x,
                                                d.x.data(),
                                                d.y.data(),
                                                d.z.data(),
                                                d.vx.data(),
                                                d.vy.data(),
                                                d.vz.data(),
                                                d.ax.data(),
                                                d.ay.data(),
                                                d.az.data(),
                                                d.m.data());
    d2Q_local[QIdx::yy] = d2QuadpoleMomentum<T>(first,
                                                last,
                                                Dim::y,
                                                Dim::y,
                                                d.x.data(),
                                                d.y.data(),
                                                d.z.data(),
                                                d.vx.data(),
                                                d.vy.data(),
                                                d.vz.data(),
                                                d.ax.data(),
                                                d.ay.data(),
                                                d.az.data(),
                                                d.m.data());
    d2Q_local[QIdx::zz] = d2QuadpoleMomentum<T>(first,
                                                last,
                                                Dim::z,
                                                Dim::z,
                                                d.x.data(),
                                                d.y.data(),
                                                d.z.data(),
                                                d.vx.data(),
                                                d.vy.data(),
                                                d.vz.data(),
                                                d.ax.data(),
                                                d.ay.data(),
                                                d.az.data(),
                                                d.m.data());
    d2Q_local[QIdx::xy] = d2QuadpoleMomentum<T>(first,
                                                last,
                                                Dim::x,
                                                Dim::y,
                                                d.x.data(),
                                                d.y.data(),
                                                d.z.data(),
                                                d.vx.data(),
                                                d.vy.data(),
                                                d.vz.data(),
                                                d.ax.data(),
                                                d.ay.data(),
                                                d.az.data(),
                                                d.m.data());
    d2Q_local[QIdx::xz] = d2QuadpoleMomentum<T>(first,
                                                last,
                                                Dim::x,
                                                Dim::z,
                                                d.x.data(),
                                                d.y.data(),
                                                d.z.data(),
                                                d.vx.data(),
                                                d.vy.data(),
                                                d.vz.data(),
                                                d.ax.data(),
                                                d.ay.data(),
                                                d.az.data(),
                                                d.m.data());
    d2Q_local[QIdx::yz] = d2QuadpoleMomentum<T>(first,
                                                last,
                                                Dim::y,
                                                Dim::z,
                                                d.x.data(),
                                                d.y.data(),
                                                d.z.data(),
                                                d.vx.data(),
                                                d.vy.data(),
                                                d.vz.data(),
                                                d.ax.data(),
                                                d.ay.data(),
                                                d.az.data(),
                                                d.m.data());

    int              rootRank = 0;
    std::array<T, 6> d2Q_global;
    MPI_Reduce(d2Q_local.data(), d2Q_global.data(), 6, MpiType<T>{}, MPI_SUM, rootRank, MPI_COMM_WORLD);

    T httplus;
    T httcross;

    computeHtt(d2Q_global, viewTheta, viewPhi, &httplus, &httcross);

    return std::make_tuple(
        httplus, httcross, d2Q_global[0], d2Q_global[1], d2Q_global[2], d2Q_global[3], d2Q_global[4], d2Q_global[5]);
}

//! @brief Observables that includes times, energies, gravitational radiation and the second derivative of the
//! quadrupole moment
template<class Dataset>
class GravWaves : public IObservables<Dataset>
{
    using T = typename Dataset::RealType;
    std::ofstream& constantsFile;
    T              viewTheta;
    T              viewPhi;

public:
    GravWaves(std::ofstream& constPath, T theta, T phi)
        : constantsFile(constPath)
        , viewTheta(theta)
        , viewPhi(phi)
    {
    }

    void computeAndWrite(Dataset& d, size_t firstIndex, size_t lastIndex, cstone::Box<T>& box)
    {

        auto [httplus, httcross, d2xx, d2yy, d2zz, d2xy, d2xz, d2yz] =
            gravRad(d, firstIndex, lastIndex, viewTheta, viewPhi);

        int rank;
        MPI_Comm_rank(d.comm, &rank);

        if (rank == 0)
        {
            fileutils::writeColumns(constantsFile,
                                    ' ',
                                    d.iteration,
                                    d.ttot,
                                    d.minDt,
                                    d.etot,
                                    d.ecin,
                                    d.eint,
                                    d.egrav,
                                    httplus,
                                    httcross,
                                    d2xx,
                                    d2yy,
                                    d2zz,
                                    d2xy,
                                    d2xz,
                                    d2yz);
        }
    }
};
} // namespace sphexa
