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
#include <cmath>

#include "cstone/sfc/box.hpp"

#include "isim_init.hpp"
#include "grid.hpp"

namespace sphexa
{

template<class Dataset>
void initHydrostaticCubeFields(Dataset& d, const std::map<std::string, double>& constants, double massPart)
{
    using T = typename Dataset::RealType;

    T r      = constants.at("r");
    T rDelta = constants.at("rDelta");

    T rhoInt = constants.at("rhoInt");
    T rhoExt = constants.at("rhoExt");

    size_t ng0  = 100;
    T      hInt = 0.5 * std::pow(3. * ng0 * massPart / 4. / M_PI / rhoInt, 1. / 3.);
    T      hExt = 0.5 * std::pow(3. * ng0 * massPart / 4. / M_PI / rhoExt, 1. / 3.);

    T pIsobaric = constants.at("pIsobaric");
    T gamma     = constants.at("gamma");

    T uInt = pIsobaric / (gamma - 1.) / rhoInt;
    T uExt = pIsobaric / (gamma - 1.) / rhoExt;

    T firstTimeStep = constants.at("firstTimeStep");

    std::fill(d.m.begin(), d.m.end(), massPart);
    std::fill(d.du_m1.begin(), d.du_m1.end(), 0.0);
    std::fill(d.mui.begin(), d.mui.end(), 10.0);
    std::fill(d.dt.begin(), d.dt.end(), firstTimeStep);
    std::fill(d.dt_m1.begin(), d.dt_m1.end(), firstTimeStep);
    std::fill(d.alpha.begin(), d.alpha.end(), d.alphamin);
    d.minDt = firstTimeStep;

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < d.x.size(); i++)
    {
        bool externalPart = (abs(d.x[i]) > r) || (abs(d.y[i]) > r) || (abs(d.z[i]) > r);

        d.h[i] = externalPart ? hExt : hInt;

        d.u[i] = externalPart ? uExt : uInt;

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
    return {{"r", .25},
            {"rDelta", .25},
            {"dim", 3},
            {"gamma", 5.0 / 3.0},
            {"rhoExt", 1.},
            {"rhoInt", 8.},
            {"pIsobaric", 2.5},         // pIsobaric = (gamma − 1.) * rho * u
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
        using T = typename Dataset::RealType;

        size_t nIntPart = cubeSide * cubeSide * cubeSide;
        d.numParticlesGlobal = nIntPart;
        auto [first, last] = partitionRange(d.numParticlesGlobal, rank, numRanks);
        resize(d, last - first);

        T r = constants_.at("r");
        regularGrid(r, cubeSide, first, last, d.x, d.y, d.z);

        T MCxInt = 0.;
        T MCyInt = 0.;
        T MCzInt = 0.;
        // Calculate mass center of the internal cube
        for (size_t i = 0; i < nIntPart; i++)
        {
            MCxInt += d.x[i];
            MCyInt += d.y[i];
            MCzInt += d.z[i];
        }
        MCxInt /= nIntPart;
        MCyInt /= nIntPart;
        MCzInt /= nIntPart;

        T stepRatio = constants_.at("rhoInt") / constants_.at("rhoExt");
        T stepInt   = (2. * r) / cubeSide;
        T stepExt   = stepInt * std::pow(stepRatio, 1./3.);

        T rDelta    = constants_.at("rDelta");
        T initR     = -(r + rDelta);
        T totalSide = 2. * (r + rDelta);

        size_t extCubeSide = round(totalSide / stepExt);

        T      totalVolume  = totalSide * totalSide * totalSide;
        size_t totalCubeExt = extCubeSide * extCubeSide * extCubeSide;
        T      massPart     = totalVolume / totalCubeExt;

        // Count additional particles
        size_t nExtPart = 0;
        T      MCxExt   = 0.;
        T      MCyExt   = 0.;
        T      MCzExt   = 0.;
        for (size_t i = 0; i < extCubeSide; i++)
        {
            T lz = initR + (i * stepExt);

            for (size_t j = 0; j < extCubeSide; j++)
            {
                T ly = initR + (j * stepExt);

                for (size_t k = 0; k < extCubeSide; k++)
                {
                    T lx = initR + (k * stepExt);

                    if ( (abs(lx) > r) || (abs(ly) > r) || (abs(lz) > r) )
                    {
                        nExtPart++;

                        MCxExt += lx;
                        MCyExt += ly;
                        MCzExt += lz;
                    }
                }
            }
        }
        MCxExt /= nExtPart;
        MCyExt /= nExtPart;
        MCzExt /= nExtPart;

        // Reside ParticleData
        d.numParticlesGlobal += nExtPart;
        resize(d, d.numParticlesGlobal);

        // Add external cube positions
        size_t idx = nIntPart;
        for (size_t i = 0; i < extCubeSide; i++)
        {
            T lz = initR + (i * stepExt);
            for (size_t j = 0; j < extCubeSide; j++)
            {
                T ly = initR + (j * stepExt);
                for (size_t k = 0; k < extCubeSide; k++)
                {
                    T lx = initR + (k * stepExt);
                    if ( (abs(lx) > r) || (abs(ly) > r) || (abs(lz) > r) )
                    {
                        d.x[idx] = lx + (MCxInt - MCxExt);
                        d.y[idx] = ly + (MCyInt - MCyExt);
                        d.z[idx] = lz + (MCzInt - MCzExt);

                        idx++;
                    }
                }
            }
        }

        initHydrostaticCubeFields(d, constants_, massPart);

        return cstone::Box<T>(-(r + (1.1 * rDelta)), r + (1.1 * rDelta), true);
    }

    const std::map<std::string, double>& constants() const override { return constants_; }
};

} // namespace sphexa
