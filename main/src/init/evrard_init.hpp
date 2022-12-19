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
 * @brief Evrard collapse initialization
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <map>

#include "cstone/sfc/box.hpp"
#include "sph/eos.hpp"

#include "isim_init.hpp"
#include "early_sync.hpp"
#include "grid.hpp"

namespace sphexa
{

std::map<std::string, double> evrardConstants()
{
    return {{"G", 1.},  {"r", 1.}, {"mTotal", 1.}, {"gamma", 5. / 3.}, {"u0", 0.05}, {"firstTimeStep", 1e-4},
            {"mui", 10}};
}

template<class Dataset>
void initEvrardFields(Dataset& d, const std::map<std::string, double>& constants)
{
    using T = typename Dataset::RealType;

    int    ng0           = 100;
    double mPart         = constants.at("mTotal") / d.numParticlesGlobal;
    double firstTimeStep = constants.at("firstTimeStep");

    d.gamma    = constants.at("gamma");
    d.muiConst = constants.at("mui");
    d.minDt    = firstTimeStep;
    d.minDt_m1 = firstTimeStep;

    std::fill(d.m.begin(), d.m.end(), mPart);
    std::fill(d.du_m1.begin(), d.du_m1.end(), 0.0);
    std::fill(d.mui.begin(), d.mui.end(), d.muiConst);
    std::fill(d.alpha.begin(), d.alpha.end(), d.alphamin);

    std::fill(d.vx.begin(), d.vx.end(), 0.0);
    std::fill(d.vy.begin(), d.vy.end(), 0.0);
    std::fill(d.vz.begin(), d.vz.end(), 0.0);

    std::fill(d.x_m1.begin(), d.x_m1.end(), 0.0);
    std::fill(d.y_m1.begin(), d.y_m1.end(), 0.0);
    std::fill(d.z_m1.begin(), d.z_m1.end(), 0.0);

    auto cv    = sph::idealGasCv(d.muiConst, d.gamma);
    auto temp0 = constants.at("u0") / cv;
    std::fill(d.temp.begin(), d.temp.end(), temp0);

    T totalVolume = 4 * M_PI / 3 * std::pow(constants.at("r"), 3);
    // before the contraction with sqrt(r), the sphere has a constant particle concentration of Ntot / Vtot
    // after shifting particles towards the center by factor sqrt(r), the local concentration becomes
    // c(r) = 2/3 * 1/r * Ntot / Vtot
    T c0 = 2. / 3. * d.numParticlesGlobal / totalVolume;

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < d.x.size(); i++)
    {
        T radius        = std::sqrt((d.x[i] * d.x[i]) + (d.y[i] * d.y[i]) + (d.z[i] * d.z[i]));
        T concentration = c0 / radius;
        d.h[i]          = std::cbrt(3 / (4 * M_PI) * ng0 / concentration) * 0.5;
    }
}

template<class Vector>
void contractRhoProfile(Vector& x, Vector& y, Vector& z)
{
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < x.size(); i++)
    {
        auto radius0 = std::sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]);

        // multiply coordinates by sqrt(r) to generate a density profile ~ 1/r
        auto contraction = std::sqrt(radius0);
        x[i] *= contraction;
        y[i] *= contraction;
        z[i] *= contraction;
    }
}

template<class Dataset>
class EvrardGlassSphere : public ISimInitializer<Dataset>
{
    std::string                   glassBlock;
    std::map<std::string, double> constants_;

public:
    EvrardGlassSphere(std::string initBlock)
        : glassBlock(initBlock)
    {
        constants_ = evrardConstants();
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

        size_t multiplicity = std::rint(cbrtNumPart / std::cbrt(blockSize));

        d.g = constants_.at("G");
        T r = constants_.at("r");

        cstone::Box<T> globalBox(-r, r, cstone::BoundaryType::open);

        unsigned level             = cstone::log8ceil<KeyType>(100 * numRanks);
        auto     initialBoundaries = cstone::initialDomainSplits<KeyType>(numRanks, level);
        KeyType  keyStart          = initialBoundaries[rank];
        KeyType  keyEnd            = initialBoundaries[rank + 1];

        assembleCube<T>(keyStart, keyEnd, globalBox, multiplicity, xBlock, yBlock, zBlock, d.x, d.y, d.z);
        cutSphere(r, d.x, d.y, d.z);

        d.numParticlesGlobal = d.x.size();
        MPI_Allreduce(MPI_IN_PLACE, &d.numParticlesGlobal, 1, MpiType<size_t>{}, MPI_SUM, simData.comm);

        contractRhoProfile(d.x, d.y, d.z);
        syncCoords<KeyType>(rank, numRanks, d.numParticlesGlobal, d.x, d.y, d.z, globalBox);

        d.resize(d.x.size());
        initEvrardFields(d, constants_);

        return globalBox;
    }

    const std::map<std::string, double>& constants() const override { return constants_; }
};

} // namespace sphexa
