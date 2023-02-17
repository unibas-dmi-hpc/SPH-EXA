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
#include "sph/eos.hpp"

#include "io/mpi_file_utils.hpp"
#include "isim_init.hpp"
#include "grid.hpp"

namespace sphexa
{

std::map<std::string, double> TurbulenceConstants()
{
    return {{"solWeight", 0.5},      {"stMaxModes", 100000}, {"Lbox", 1.0},       {"stMachVelocity", 0.3e0},
            {"firstTimeStep", 1e-4}, {"epsilon", 1e-15},     {"rngSeed", 251299}, {"stSpectForm", 1},
            {"mTotal", 1.0},         {"powerLawExp", 5 / 3}, {"anglesExp", 2.0},  {"gamma", 1.001},
            {"mui", 0.62},           {"u0", 1000.},          {"Kcour", 0.4}};
}

template<class Dataset>
void initTurbulenceHydroFields(Dataset& d, const std::map<std::string, double>& constants)
{
    size_t ng0           = 100;
    double mPart         = constants.at("mTotal") / d.numParticlesGlobal;
    double Lbox          = constants.at("Lbox");
    double hInit         = std::cbrt(3.0 / (4. * M_PI) * ng0 * std::pow(Lbox, 3) / d.numParticlesGlobal) * 0.5;
    double firstTimeStep = constants.at("firstTimeStep");

    d.gamma    = constants.at("gamma");
    d.muiConst = constants.at("mui");
    d.Kcour    = constants.at("Kcour");
    d.minDt    = firstTimeStep;
    d.minDt_m1 = firstTimeStep;

    auto cv    = sph::idealGasCv(d.muiConst, d.gamma);
    auto temp0 = constants.at("u0") / cv;

    std::fill(d.m.begin(), d.m.end(), mPart);
    std::fill(d.du_m1.begin(), d.du_m1.end(), 0.0);
    std::fill(d.h.begin(), d.h.end(), hInit);
    std::fill(d.mui.begin(), d.mui.end(), d.muiConst);
    std::fill(d.alpha.begin(), d.alpha.end(), d.alphamin);
    std::fill(d.temp.begin(), d.temp.end(), temp0);

    std::fill(d.vx.begin(), d.vx.end(), 0.);
    std::fill(d.vy.begin(), d.vy.end(), 0.);
    std::fill(d.vz.begin(), d.vz.end(), 0.);
    std::fill(d.x_m1.begin(), d.x_m1.end(), 0.);
    std::fill(d.y_m1.begin(), d.y_m1.end(), 0.);
    std::fill(d.z_m1.begin(), d.z_m1.end(), 0.);
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

    cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, size_t cbrtNumPart,
                                                 Dataset& simData) const override
    {
        auto& d       = simData.hydro;
        using KeyType = typename Dataset::KeyType;
        using T       = typename Dataset::RealType;

        std::vector<T> xBlock, yBlock, zBlock;
        fileutils::readTemplateBlock(glassBlock, xBlock, yBlock, zBlock);
        size_t blockSize = xBlock.size();

        size_t                    multi1D      = std::rint(cbrtNumPart / std::cbrt(blockSize));
        std::tuple<int, int, int> multiplicity = {multi1D, multi1D, multi1D};
        d.numParticlesGlobal                   = multi1D * multi1D * multi1D * blockSize;

        cstone::Box<T> globalBox(-0.5, 0.5, cstone::BoundaryType::periodic);

        unsigned level             = cstone::log8ceil<KeyType>(100 * numRanks);
        auto     initialBoundaries = cstone::initialDomainSplits<KeyType>(numRanks, level);
        KeyType  keyStart          = initialBoundaries[rank];
        KeyType  keyEnd            = initialBoundaries[rank + 1];

        assembleRectangle<T>(keyStart, keyEnd, globalBox, multiplicity, xBlock, yBlock, zBlock, d.x, d.y, d.z);

        d.resize(d.x.size());

        initTurbulenceHydroFields(d, constants_);

        return globalBox;
    }

    const std::map<std::string, double>& constants() const override { return constants_; }
};

} // namespace sphexa
