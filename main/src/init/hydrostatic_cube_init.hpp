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
 * @brief Hydrostatic cube simulation data initialization
 *
 * @author Jose A. Escartin <ja.escartin@gmail.com>"
 */

#pragma once

#include <map>

#include "cstone/sfc/box.hpp"

#include "isim_init.hpp"
#include "grid.hpp"

namespace sphexa
{

template<class Dataset>
void initHydrostaticCubeFields(Dataset& d, const std::map<std::string, double>& constants)
{
    using T = typename Dataset::RealType;

    // How to select the h with different densities?
    double r             = constants.at("r");
    double rDelta        = constants.at("rDelta");

    double totalVolume   = 8.0 * r * r * r;
    int    ng0           = 100;
    double hInit         = std::cbrt(3.0 / (4 * M_PI) * ng0 * totalVolume / d.numParticlesGlobal) * 0.5;
    std::fill(d.h.begin(), d.h.end(), hInit);

    double mPart         = constants.at("mTotal") / d.numParticlesGlobal;
    double firstTimeStep = constants.at("firstTimeStep");
    double uExt          = constants.at("pIsobaric")/(constants.at("gamma")-1.)/constants.at("rhoExt");
    double uInt          = constants.at("pIsobaric")/(constants.at("gamma")-1.)/constants.at("rhoInt");

    std::fill(d.m.begin(), d.m.end(), mPart);
    std::fill(d.du_m1.begin(), d.du_m1.end(), 0.0);
    std::fill(d.mui.begin(), d.mui.end(), 10.0);
    std::fill(d.dt.begin(), d.dt.end(), firstTimeStep);
    std::fill(d.dt_m1.begin(), d.dt_m1.end(), firstTimeStep);
    std::fill(d.alpha.begin(), d.alpha.end(), d.alphamin);
    d.minDt = firstTimeStep;

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < d.x.size(); i++)
    {
        bool ext = d.x[i] < r || d.y[i] < r || d.z[i] < r || d.x[i] > r || d.y[i] > r || d.z[i] > r;

        d.u[i] = ext ? uExt : uInt;

        d.vx[i] = 0.;
        d.vy[i] = 0.;
        d.vz[i] = 0.;

        d.x_m1[i] = d.x[i] - d.vx[i] * firstTimeStep;
        d.y_m1[i] = d.y[i] - d.vy[i] * firstTimeStep;
        d.z_m1[i] = d.z[i] - d.vz[i] * firstTimeStep;
    }
}

std::map<std::string, double> HydrostaticCubeConstants()
{
    return {{"r", 0.5},
            {"rDelta", 0.25},
            {"mTotal", 1.},
            {"dim", 3},
            {"gamma", 5.0 / 3.0},
            {"rhoExt", 1.},
            {"rhoInt", 4.},
            {"pIsobaric", 2.5},     // pIsobaric = (gamma − 1)*rho*u
            {"firstTimeStep", 1e-4}};
}

template<class Dataset>
class HydrostaticCubeGrid : public ISimInitializer<Dataset>
{
    std::map<std::string, double> constants_;

public:
    HydrostaticCubeGrid() { constants_ = HydrostaticCubeConstants(); }

    cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, size_t cubeSide, Dataset& d) const override
    {
        using T              = typename Dataset::RealType;
        d.numParticlesGlobal = cubeSide * cubeSide * cubeSide;

        auto [first, last] = partitionRange(d.numParticlesGlobal, rank, numRanks);
        resize(d, last - first);

        T r      = constants_.at("r")
        T rDelta = constants_.at("rDelta");
        T rhoInt = constants_.at("rhoInt");
        T rhoExt = constants_.at("rhoExt");

        internalCubeGrid(r, rDelta, rhoInt, rhoExt, cubeSide, first, last, d.x, d.y, d.z);

        initHydrostaticCubeFields(d, constants_);

        return cstone::Box<T>(-r, r, false);
    }

    const std::map<std::string, double>& constants() const override { return constants_; }
};

template<class Dataset>
class HydrostaticCubeGlass : public ISimInitializer<Dataset>
{
    std::string                   glassBlock;
    std::map<std::string, double> constants_;

public:
    HydrostaticCubeGlass(std::string initBlock)
        : glassBlock(initBlock)
    {
        constants_ = HydrostaticCubeConstants();
    }

    cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, size_t cbrtNumPart, Dataset& d) const override
    {
        using KeyType = typename Dataset::KeyType;
        using T       = typename Dataset::RealType;

        std::vector<T> xBlock, yBlock, zBlock;
        fileutils::readTemplateBlock(glassBlock, xBlock, yBlock, zBlock);
        size_t blockSize = xBlock.size();

        size_t multiplicity = std::rint(cbrtNumPart / std::cbrt(blockSize));

        T              r = constants_.at("r1");
        cstone::Box<T> globalBox(-r, r, false);

        auto [keyStart, keyEnd] = partitionRange(cstone::nodeRange<KeyType>(0), rank, numRanks);
        assembleCube<T>(keyStart, keyEnd, globalBox, multiplicity, xBlock, yBlock, zBlock, d.x, d.y, d.z);
        cutSphere(r, d.x, d.y, d.z);

        d.numParticlesGlobal = d.x.size();
        MPI_Allreduce(MPI_IN_PLACE, &d.numParticlesGlobal, 1, MpiType<size_t>{}, MPI_SUM, d.comm);

        resize(d, d.x.size());

        double totalVolume = 4. * M_PI / 3. * r * r * r;
        initHydrostaticCubeFields(d, totalVolume, constants_);

        return globalBox;
    }

    const std::map<std::string, double>& constants() const override { return constants_; }
};

} // namespace sphexa
