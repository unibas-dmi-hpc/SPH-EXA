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
 * @brief Turbulence simulation data initialization
 *
 * @author Axel Sanz <axelsanzlechuga@gmail.com>
 */

#pragma once

#include <map>
#include <cmath>
#include <algorithm>

#include "cstone/sfc/box.hpp"

#include "isim_init.hpp"
#include "grid.hpp"

#include "sph/hydro_turb/turbulence_data.hpp"
#include "sph/hydro_turb/create_modes.hpp"

namespace sphexa
{

std::map<std::string, double> TurbulenceConstants()
{
    return {{"stSolWeight", 0.5},
            {"stMaxModes", 100000},
            {"Lbox", 1.0},
            {"stMachVelocity", 0.3e0},
            {"dim", 3},
            {"firstTimeStep", 1e-4},
            {"epsilon", 1e-15},
            {"stSeedIni", 251299},
            {"stSpectForm", 2},
            {"mTotal", 1.0},
            {"powerLawExp", 5 / 3},
            {"anglesExp", 2.0}};
}

template<class Dataset>
void initTurbulenceHydroFields(Dataset& d, const std::map<std::string, double>& constants)
{
    size_t ng0           = 100;
    double mPart         = constants.at("mTotal") / d.numParticlesGlobal;
    double Lbox          = constants.at("Lbox");
    double hInit         = std::cbrt(3.0 / (4. * M_PI) * ng0 * std::pow(Lbox, 3) / d.numParticlesGlobal) * 0.5;
    double firstTimeStep = constants.at("firstTimeStep");

    std::fill(d.m.begin(), d.m.end(), mPart);
    std::fill(d.du_m1.begin(), d.du_m1.end(), 0.0);
    std::fill(d.h.begin(), d.h.end(), hInit);
    std::fill(d.mui.begin(), d.mui.end(), 10.0);
    std::fill(d.alpha.begin(), d.alpha.end(), d.alphamin);
    std::fill(d.u.begin(), d.u.end(), 1000.0);

    d.minDt    = firstTimeStep;
    d.minDt_m1 = firstTimeStep;

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < d.x.size(); i++)
    {
        d.vx[i]   = 0.;
        d.vy[i]   = 0.;
        d.vz[i]   = 0.;
        d.x_m1[i] = d.x[i] - d.vx[i] * firstTimeStep;
        d.y_m1[i] = d.y[i] - d.vy[i] * firstTimeStep;
        d.z_m1[i] = d.z[i] - d.vz[i] * firstTimeStep;
    }
}

template<class T>
void initTurbulenceModes(sph::TurbulenceData<T>& turb, const std::map<std::string, double>& constants)
{
    double   eps         = constants.at("epsilon");
    size_t   stMaxModes  = constants.at("stMaxModes");
    double   Lbox        = constants.at("Lbox");
    double   velocity    = constants.at("stMachVelocity");
    long int seed        = constants.at("stSeedIni");
    size_t   stSpectForm = constants.at("stSpectForm");
    double   powerLawExp = constants.at("powerLawExp");
    double   anglesExp   = constants.at("anglesExp");

    double twopi     = 2.0 * M_PI;
    double stEnergy  = 5.0e-3 * std::pow(velocity, 3) / Lbox;
    double stStirMin = (1.0 - eps) * twopi / Lbox;
    double stStirMax = (3.0 + eps) * twopi / Lbox;

    turb.numDim      = constants.at("dim");
    turb.decayTime   = Lbox / (2.0 * velocity);
    turb.stSolWeight = constants.at("stSolWeight");
    turb.stSeed      = seed;

    turb.amplitudes.resize(stMaxModes);
    turb.modes.resize(stMaxModes * turb.numDim);

    sph::createStirringModes(turb,
                             Lbox,
                             Lbox,
                             Lbox,
                             stMaxModes,
                             stEnergy,
                             stStirMax,
                             stStirMin,
                             turb.numDim,
                             turb.stSeed,
                             stSpectForm,
                             powerLawExp,
                             anglesExp);

    std::cout << "Total Number of Stirring Modes: " << turb.numModes << std::endl;
    turb.amplitudes.resize(turb.numModes);
    turb.modes.resize(turb.numModes * turb.numDim);
    turb.phases.resize(6 * turb.numModes);

    sph::fillRandomGaussian(turb.phases, turb.variance, turb.stSeed);
}

template<class Dataset>
class TurbulenceGlass : public ISimInitializer<Dataset>
{
    std::string                   glassBlock;
    std::map<std::string, double> constants_;

public:
    TurbulenceGlass(std::string initBlock)
        : glassBlock(initBlock)
    {
        constants_ = TurbulenceConstants();
    }

    cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, size_t cbrtNumPart, Dataset& d) const override
    {
        using KeyType = typename Dataset::KeyType;
        using T       = typename Dataset::RealType;

        std::vector<T> xBlock, yBlock, zBlock;
        fileutils::readTemplateBlock(glassBlock, xBlock, yBlock, zBlock);
        size_t blockSize = xBlock.size();

        size_t multiplicity  = std::rint(cbrtNumPart / std::cbrt(blockSize));
        d.numParticlesGlobal = multiplicity * multiplicity * multiplicity * blockSize;

        cstone::Box<T> globalBox(-0.5, 0.5, true);
        auto [keyStart, keyEnd] = partitionRange(cstone::nodeRange<KeyType>(0), rank, numRanks);
        assembleCube<T>(keyStart, keyEnd, globalBox, multiplicity, xBlock, yBlock, zBlock, d.x, d.y, d.z);

        d.resize(d.x.size());

        initTurbulenceModes(d.turbulenceData, constants_);
        initTurbulenceHydroFields(d, constants_);

        return globalBox;
    }

    const std::map<std::string, double>& constants() const override { return constants_; }
};

} // namespace sphexa
