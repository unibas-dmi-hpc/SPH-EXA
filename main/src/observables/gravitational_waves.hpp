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

/*! @file calculate gravitational waves observable
 *
 *  @author Lukas Schmidt
 */

#include <array>
#include "mpi.h"

#include "sph/math.hpp"
#include "iobservables.hpp"
#include "io/ifile_writer.hpp"

namespace sphexa
{

//!@brief calculates the second derivative of the quadpole momentum
template<class T, class Dataset>
T d2QuadpoleMomentum(size_t begin, size_t end, T* pos1, T* pos2, T* vel1, T* vel2, T* gradp1, T* gradp2, T* m, Dataset& d)
{
    T out;
    if(pos1 == pos2)
    {
        T* x        = d.x.data();
        T* y        = d.y.data();
        T* z        = d.z.data();
        T* vx       = d.vx.data();
        T* vy       = d.vy.data();
        T* vz       = d.vz.data();
        T* gradP_x  = d.grad_P_x.data();
        T* gradP_y  = d.grad_P_y.data();
        T* gradP_z  = d.grad_P_z.data();

#pragma omp parallel for reduction(+ : out)
        for(size_t i = begin; i < end; i++)
        {
            T acc1 = -gradp1[i];
            T acc2 = -gradp2[i];
            T factor1 = vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i];
            T factor2 = x[i] * (-gradP_x[i]) + y[i] * (-gradP_y[i]) + z[i] * (-gradP_z[i]);

            out += (3.0 * (vel1[i] * vel1[i] + pos1[i] * acc1) - factor1 - factor2) * m[i];

        }
        return out * 2.0 / 3.0;
    }
    else
    {
#pragma omp parallel for reduction(+ : out)
        for(size_t i = begin; i < end; i++)
        {
            T acc1 = -gradp1[i];
            T acc2 = -gradp2[i];
            out += (2.0 * vel1[i] * vel2[i] + pos1[i] * acc2 + pos2[i] * acc1) * m[i];
        }
        return out;
    }
}

/*!@brief calculates the gravitational waves observable
 *
 * @tparam T            floating point type
 * @tparam Dataset
 * @param d             Dataset
 * @param first         first locally assigned particle index of buffers in @p d
 * @param last          first locally assigned particle index of buffers in @p d
 * @param viewTheta     viewing angle to determine ...
 * @param viewPhi       viewing angle to determine ...
 * @return              array containing the observables and the second derivative of the quadpole momentum
 */
template<class T, class Dataset>
std::array<T, 8> gravRad(Dataset& d,size_t first, size_t last, T viewTheta, T viewPhi)
{
    T dot2ibartt;
    T dot2ibarpp;
    T dot2ibartp;

    T g = 6.6726e-8;    //Gravitational constant
    T c = 2.997924562e10;   //Speed of light
    T gwunits = std::pow(g/c, 4.0/3.08568025e22); //Gravitational Waves unit at 10kpc

    std::array<T, 6> d2Local;

    d2Local[0] = d2QuadpoleMomentum(first, last, d.x.data(), d.x.data(),
                                    d.vx.data(), d.vx.data(), d.grad_P_x.data(), d.grad_P_x.data(), d.m.data(), d);
    d2Local[1] = d2QuadpoleMomentum(first, last, d.y.data(), d.y.data(),
                                    d.vy.data(), d.vy.data(), d.grad_P_y.data(), d.grad_P_y.data(), d.m.data(), d);
    d2Local[2] = d2QuadpoleMomentum(first, last, d.z.data(), d.z.data(),
                                    d.vz.data(), d.vz.data(), d.grad_P_z.data(), d.grad_P_z.data(), d.m.data(), d);

    d2Local[3] = d2QuadpoleMomentum(first, last, d.x.data(), d.y.data(),
                                    d.vx.data(), d.vy.data(), d.grad_P_x.data(), d.grad_P_y.data(), d.m.data(), d);
    d2Local[4] = d2QuadpoleMomentum(first, last, d.x.data(), d.z.data(),
                                    d.vx.data(), d.vz.data(), d.grad_P_x.data(), d.grad_P_z.data(), d.m.data(), d);
    d2Local[5] = d2QuadpoleMomentum(first, last, d.y.data(), d.z.data(),
                                    d.vy.data(), d.vz.data(), d.grad_P_y.data(), d.grad_P_z.data(), d.m.data(), d);


    int rootRank = 0;
    std::array<T, 6> d2Global;
    MPI_Reduce(d2Local.data(), d2Global.data(),6,MpiType<T>{}, MPI_SUM, rootRank, MPI_COMM_WORLD);

    if (viewTheta == 0.0 && viewPhi == 0.0)
    {
        dot2ibartt = d2Global[0];
        dot2ibarpp = d2Global[1];
        dot2ibartp = d2Global[3];
    }
    else
    {
        dot2ibartt = (d2Global[0] * std::pow(std::cos(viewPhi), 2) + d2Global[1] * std::pow(std::sin(viewPhi), 2) + d2Global[3] * std::sin(2.0 * viewPhi)
                      * std::pow(std::cos(viewTheta), 2) + d2Global[2] * std::pow(std::sin(viewTheta), 2)
                      - (d2Global[4] * std::cos(viewPhi) + d2Global[5] * std::sin(viewPhi)) * std::sin(2.0 * viewTheta));

        dot2ibarpp = d2Global[0] * std::pow(std::cos(viewPhi), 2) + d2Global[1] * std::pow(std::cos(viewPhi), 2) - d2Global[3] * std::sin(2.0 * viewPhi);

        dot2ibartp = 0.5 * (d2Global[1] - d2Global[0]) * std::cos(viewTheta) * std::sin(2.0 * viewPhi) + d2Global[3] * std::cos(viewTheta) * std::cos(2.0 * viewPhi)
                        + (d2Global[4] * std::sin(viewPhi) - d2Global[5] * std::cos(viewPhi)) * std::sin(viewTheta);
    }

    T httplus = (dot2ibartt - dot2ibarpp) * gwunits;
    T httcross = 2.0 * dot2ibartp * gwunits;

    return {httplus, httcross, d2Global[0], d2Global[1], d2Global[2], d2Global[3], d2Global[4], d2Global[5]};
}


//! @brief Observables that includes times, energies and gravitational wave something //TODO
template<class Dataset>
class GravWaves : public IObservables<Dataset>
{
    std::ofstream& constantsFile;

public:
    GravWaves(std::ofstream& constPath)
        : constantsFile(constPath)
    {
    }

    using T = typename Dataset::RealType;

    void computeAndWrite(Dataset& d, size_t firstIndex, size_t lastIndex, cstone::Box<T>& box)
    {
        T viewTheta = 0.0;
        T viewPhi = 0.0;

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