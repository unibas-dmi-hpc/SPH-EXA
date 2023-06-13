/*
 * MIT License
 *
 * Copyright (c) 2022 CSCS, ETH Zurich
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

#include <mpi.h>

#include "io/file_utils.hpp"
#include "conserved_quantities.hpp"
#include "gpu_reductions.h"
#include "iobservables.hpp"
#include "power_spectrum_calculations.hpp"

namespace sphexa
{

template<class Dataset>
double localMachSquareSum(size_t first, size_t last, Dataset& d)
{
    const auto* vx = d.vx.data();
    const auto* vy = d.vy.data();
    const auto* vz = d.vz.data();
    const auto* c  = d.c.data();

    double localMachSquareSum = 0.0;

#pragma omp parallel for reduction(+ : localMachSquareSum)
    for (size_t i = first; i < last; ++i)
    {
        util::array<double, 3> V{vx[i], vy[i], vz[i]};
        localMachSquareSum += norm2(V) / (c[i] * c[i]);
    }

    return localMachSquareSum;
}

/*!@brief global calculation of the Mach number RMS
 *
 * @tparam Dataset
 * @param[in]     first     first locally assigned particle index of buffers in @p d
 * @param[in]     last      last locally assigned particle index of buffers in @p d
 * @param d                 particle dataset
 * @param comm              MPI communication rank
 * @return
 */
template<class Dataset>
double calculateMachRMS(size_t first, size_t last, Dataset& d, MPI_Comm comm)
{
    double localMachRms;

    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        localMachRms = machSquareSumGpu(rawPtr(d.devData.vx), rawPtr(d.devData.vy), rawPtr(d.devData.vz),
                                        rawPtr(d.devData.c), first, last);
    }
    else { localMachRms = localMachSquareSum(first, last, d); }

    int    rootRank = 0;
    double globalMachRms;

    MPI_Reduce(&localMachRms, &globalMachRms, 1, MPI_DOUBLE, MPI_SUM, rootRank, comm);

    return std::sqrt(globalMachRms / d.numParticlesGlobal);
}

//! @brief Observables that includes times, energies and the root mean square of the mach number
template<class Dataset, class DomainType>
class TurbulenceMachRMS : public IObservables<Dataset, DomainType>
{
    std::ofstream& constantsFile;

public:
    TurbulenceMachRMS(std::ofstream& constPath)
        : constantsFile(constPath)
    {
    }

    using T = typename Dataset::RealType;

    void computeAndWrite(Dataset& simData, size_t firstIndex, size_t lastIndex, cstone::Box<T>& box)
    {
        auto& d = simData.hydro;
        computeConservedQuantities(firstIndex, lastIndex, d, simData.comm);
        double machRms = calculateMachRMS(firstIndex, lastIndex, d, simData.comm);

        int rank;
        MPI_Comm_rank(simData.comm, &rank);

        if (rank == 0)
        {
            fileutils::writeColumns(constantsFile, ' ', d.iteration, d.ttot, d.minDt, d.etot, d.ecin, d.eint, d.egrav,
                                    d.linmom, d.angmom, machRms);
        }
    }

    void occasionalObservation(Dataset& simData, size_t firstIndex, size_t lastIndex,
                               std::unique_ptr<Propagator<DomainType, Dataset>> prop,
                               cstone::Box<typename Dataset::RealType>&         box)
    {
        auto& d              = simData.hydro;
        using T              = typename Dataset::RealType;
        size_t dim           = std::cbrt(d.numParticlesGlobal); // each dimension size
        bool   centered_at_0 = true;

        double Lbox          = box.lx();               // length of the box
        size_t npart         = lastIndex - firstIndex; // # of real local particles
        size_t pixelsPerRank = npart * 8;              // 2xdim in each dimension decomposed into domain
        size_t npixelsPerDim = 2 * dim;
        int    Kmax          = std::ceil(std::sqrt(3) * npixelsPerDim * 0.5);
        double E[Kmax], k_center[Kmax];
        size_t numLocalParticles = d.x.size(); // # of local particles including halo
        double xpos[npart], ypos[npart], zpos[npart];

        GriddedDomain<T> gd(d.numParticlesGlobal, Lbox);

        // For testing
        std::vector<double> rho, h, v;
        rho.resize(npart);
        h.resize(npart);
        v.resize(npart);
        std::cout << "numlocalparts = " << numLocalParticles << std::endl;
        std::cout << "npart = " << npart << std::endl;

        std::cout << "first index = " << firstIndex << std::endl;
        std::cout << "last index = " << lastIndex << std::endl;

        // std::terminate();

        const auto* xptr  = &(d.x.data()[firstIndex]);
        const auto* yptr  = &(d.y.data()[firstIndex]);
        const auto* zptr  = &(d.z.data()[firstIndex]);
        const auto* hptr  = &(d.h.data()[firstIndex]);
        const auto* vxptr = &(d.vx.data()[firstIndex]);
        const auto* vyptr = &(d.vy.data()[firstIndex]);
        const auto* vzptr = &(d.vz.data()[firstIndex]);
        const auto* kxptr = &(d.kx.data()[firstIndex]);
        const auto* mptr  = &(d.m.data()[firstIndex]);
        const auto* xmptr = &(d.xm.data()[firstIndex]);

        // Move the global box into (0,1) range in every dimension from (-0.5,0.5)
        if (centered_at_0)
        {
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < npart; i++)
            {
                xpos[i]    = xptr[i] + 0.5 * (box.lx());
                ypos[i]    = yptr[i] + 0.5 * (box.ly());
                zpos[i]    = zptr[i] + 0.5 * (box.lz());
                h[i]       = hptr[i];
                double vxi = vxptr[i];
                double vyi = vyptr[i];
                double vzi = vzptr[i];
                v[i]       = std::sqrt(vxi * vxi + vyi * vyi * vzi * vzi);
                rho[i]     = kxptr[i] * mptr[i] / xmptr[i];
            }
        }

        std::cout << "gridding start" << std::endl;
        gd.rasterizeDomain(xpos, ypos, zpos, v.data(), rho.data(), h.data(), npart);
        std::cout << "gridding end" << std::endl;

        // gridding_spectra(numLocalParticles, Lbox, xpos, ypos, zpos, d.divv.data(), d.rho.data(), mass, partperpixel,
        //                  npixelsPerDim, gd.Gv.data(), E, k_center);

        // For testing, production should write to hdf5
        // std::ofstream spectra;
        // std::string   spectra_filename = "turb_spectra.txt";
        // spectra.open(spectra_filename, std::ios::trunc);
        // for (int i = 0; i < Kmax; i++)
        // {
        //     spectra << std::setprecision(8) << std::scientific << k_center[i] << ' ' << E[i] << std::endl;
        // }
        // spectra.close();

        std::cout << "spectrum end" << std::endl;
    }
};

} // namespace sphexa