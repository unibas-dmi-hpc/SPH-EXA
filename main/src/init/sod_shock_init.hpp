/*
 * MIT License
 *
 * Copyright (c) 2023 CSCS, ETH Zurich
 *               2023 University of Basel
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

/*! @file initialize Sod Shock test from a pre-relaxed initial condition
 *
 * @author Lukas Schmidt
 */

#pragma once

#include "cstone/sfc/box.hpp"
#include "cstone/sfc/sfc.hpp"
#include "cstone/primitives/gather.hpp"
#include "io/mpi_file_utils.hpp"
#include "isim_init.hpp"
#include "sph/eos.hpp"

namespace sphexa
{

template<class T, class Dataset>
void initSodShock(Dataset& d, const std::map<std::string, double>& constants, T massPart)
{
    T rhoLeft  = constants.at("rho_l");
    T rhoRight = constants.at("rho_r");
    T pLeft    = constants.at("P_l");
    T pRight   = constants.at("P_r");

    T hLeft  = 0.5 * std::cbrt(3. * d.ng0 * massPart / 4. / M_PI / rhoLeft);
    T hRight = 0.5 * std::cbrt(3. * d.ng0 * massPart / 4. / M_PI / rhoRight);

    std::fill(d.m.begin(), d.m.end(), massPart);
    std::fill(d.du_m1.begin(), d.du_m1.end(), 0.0);
    std::fill(d.mue.begin(), d.mue.end(), 2.0);
    std::fill(d.mui.begin(), d.mui.end(), 10.0);
    std::fill(d.alpha.begin(), d.alpha.end(), d.alphamin);
    std::fill(d.vx.begin(), d.vx.end(), 0.0);
    std::fill(d.vy.begin(), d.vy.end(), 0.0);
    std::fill(d.vz.begin(), d.vz.end(), 0.0);
    std::fill(d.x_m1.begin(), d.x_m1.end(), 0.0);
    std::fill(d.y_m1.begin(), d.y_m1.end(), 0.0);
    std::fill(d.z_m1.begin(), d.z_m1.end(), 0.0);

    auto cv    = sph::idealGasCv(d.muiConst, d.gamma);
    auto gamma = d.gamma;

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < d.x.size(); i++)
    {
        if (d.x[i] < 0.5)
        {
            T u       = pLeft / (gamma - 1.0) / rhoLeft;
            d.temp[i] = u / cv;
            d.h[i]    = hLeft;
        }
        else
        {
            T u       = pRight / (gamma - 1.0) / rhoRight;
            d.temp[i] = u / cv;
            d.h[i]    = hRight;
        }
    }
}

std::map<std::string, double> SodShockConstants()
{
    return {{"P_l", 1.0},       {"P_r", 0.1},    {"rho_l", 1.0},    {"rho_r", 0.125},
            {"gamma", 5. / 3.}, {"minDt", 1e-6}, {"minDt_m1", 1e-6}};
}

template<class Dataset>
class SodShockInit : public ISimInitializer<Dataset>
{
    std::string glassBlock;
    using Base = ISimInitializer<Dataset>;
    using Base::settings_;

public:
    SodShockInit(std::string initBlock)
        : glassBlock(initBlock)
    {
        Base::updateSettings(SodShockConstants());
    }

    cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, size_t cbrtNumPart,
                                                 Dataset& simData) const override
    {
        using KeyType = typename Dataset::KeyType;
        using T       = typename Dataset::RealType;
        auto& d       = simData.hydro;
        auto  pbc     = cstone::BoundaryType::periodic;
        auto  fbc     = cstone::BoundaryType::fixed;

        std::vector<T> xBlock, yBlock, zBlock;
        fileutils::readTemplateBlock(glassBlock, d.x, d.y, d.z);

        cstone::Box<T> globalBox(0, 1, 0, 0.2, 0, 0.25, fbc, pbc, pbc);

        d.numParticlesGlobal = d.x.size();
        syncCoords<KeyType>(rank, numRanks, d.numParticlesGlobal, d.x, d.y, d.z, globalBox);
        d.resize(d.x.size());

        settings_["numParticlesGlobal"] = double(d.numParticlesGlobal);
        BuiltinWriter attributeSetter(settings_);
        d.loadOrStoreAttributes(&attributeSetter);

        T rhoHigh = settings_.at("rho_l");
        T rhoLow = settings_.at("rho_r");

        T highDensVolume = globalBox.lx() * globalBox.ly() * globalBox.lz() * 0.5;
        T nPartHighDens = d.x.size() * rhoHigh / (rhoHigh+rhoLow); //estimate from template block
        T      particleMass = highDensVolume * settings_.at("rho_l") / nPartHighDens;

        initSodShock(d, settings_, particleMass);

        return globalBox;
    }

    const std::map<std::string, double>& constants() const override { return settings_; }
};

} // namespace sphexa