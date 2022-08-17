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
 * @brief Wind shock simulation data initialization
 *
 * @author Ruben Cabezon <ruben.cabezon@unibas.ch>
 * @author Jose A. Escartin <ja.escartin@gmail.com>
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 */

#pragma once

#include <map>
#include <cmath>
#include <algorithm>

#include "cstone/sfc/box.hpp"

#include "io/file_utils.hpp"
#include "isim_init.hpp"
#include "grid.hpp"

namespace sphexa
{

template<class Dataset>
void initWindShockFields(Dataset& d, const std::map<std::string, double>& constants, double massPart)
{
    using T = typename Dataset::RealType;

    T r             = constants.at("r");
    T rSphere       = constants.at("rSphere");
    T rhoInt        = constants.at("rhoInt");
    T rhoExt        = constants.at("rhoExt");
    T uExt          = constants.at("uExt");
    T vxExt         = constants.at("vxExt");
    T vyExt         = constants.at("vyExt");
    T vzExt         = constants.at("vzExt");
    T firstTimeStep = constants.at("firstTimeStep");
    T epsilon       = constants.at("epsilon");

    size_t ng0  = 100;
    T      hInt = 0.5 * std::cbrt(3. * ng0 * massPart / 4. / M_PI / rhoInt);
    T      hExt = 0.5 * std::cbrt(3. * ng0 * massPart / 4. / M_PI / rhoExt);

    std::fill(d.m.begin(), d.m.end(), massPart);
    std::fill(d.du_m1.begin(), d.du_m1.end(), 0.0);
    std::fill(d.mui.begin(), d.mui.end(), 10.0);
    std::fill(d.alpha.begin(), d.alpha.end(), d.alphamin);

    d.minDt    = firstTimeStep;
    d.minDt_m1 = firstTimeStep;

    T uInt = uExt / (rhoInt / rhoExt);

    T k = 150. / r;

    util::array<T, 3> blobCenter{r, r, r};

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < d.x.size(); i++)
    {
        util::array<T, 3> X{d.x[i], d.y[i], d.z[i]};

        T rPos = std::sqrt(norm2(X - blobCenter));

        if (rPos > rSphere + epsilon)
        {
            if (rPos > rSphere + 2 * hExt)
            {
                // more than two smoothing lengths away from the inner sphere
                d.h[i] = hExt;
            }
            else
            {
                // reduce smoothing lengths for particles outside, but close to the inner sphere
                d.h[i] = hInt + 0.5 * (hExt - hInt) * (1. + std::tanh(k * (rPos - rSphere - hExt)));
            }

            d.u[i]  = uExt;
            d.vx[i] = vxExt;
            d.vy[i] = vyExt;
            d.vz[i] = vzExt;
        }
        else
        {
            d.h[i]  = hInt;
            d.u[i]  = uInt;
            d.vx[i] = 0.;
            d.vy[i] = 0.;
            d.vz[i] = 0.;
        }

        d.x_m1[i] = d.x[i] - d.vx[i] * firstTimeStep;
        d.y_m1[i] = d.y[i] - d.vy[i] * firstTimeStep;
        d.z_m1[i] = d.z[i] - d.vz[i] * firstTimeStep;
    }
}

std::map<std::string, double> WindShockConstants()
{
    return {{"r", .125},
            {"rSphere", .025},
            {"rhoInt", 10.},
            {"rhoExt", 1.},
            {"uExt", 3. / 2.},
            {"vxExt", 2.7},
            {"vyExt", .0},
            {"vzExt", .0},
            {"dim", 3},
            {"gamma", 5. / 3.},
            {"firstTimeStep", 1e-10},
            {"epsilon", 0.}};
}

template<class Dataset>
class WindShockGlass : public ISimInitializer<Dataset>
{
    std::string                   glassBlock;
    std::map<std::string, double> constants_;

public:
    WindShockGlass(std::string initBlock)
        : glassBlock(initBlock)
    {
        constants_ = WindShockConstants();
    }

    cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, size_t cbrtNumPart, Dataset& d) const override
    {
        using KeyType = typename Dataset::KeyType;
        using T       = typename Dataset::RealType;

        T r       = constants_.at("r");
        T rSphere = constants_.at("rSphere");
        T rhoInt  = constants_.at("rhoInt");
        T rhoExt  = constants_.at("rhoExt");
        T epsilon = constants_.at("epsilon");

        std::vector<T> xBlock, yBlock, zBlock;
        fileutils::readTemplateBlock(glassBlock, xBlock, yBlock, zBlock);
        size_t blockSize = xBlock.size();

        size_t multiplicity  = std::rint(cbrtNumPart / std::cbrt(blockSize));
        d.numParticlesGlobal = multiplicity * multiplicity * multiplicity * blockSize;

        cstone::Box<T> globalBox(0, 4 * r, 0, 2 * r, 0, 2 * r, true);
        cstone::Box<T> boxA(0, 2 * r, 0, 2 * r, 0, 2 * r, true);
        cstone::Box<T> boxB(2 * r, 4 * r, 0, 2 * r, 0, 2 * r, true);

        auto [keyStart, keyEnd] = partitionRange(cstone::nodeRange<KeyType>(0), rank, numRanks);
        assembleCube<T>(keyStart, keyEnd, boxA, multiplicity, xBlock, yBlock, zBlock, d.x, d.y, d.z);
        assembleCube<T>(keyStart, keyEnd, boxB, multiplicity, xBlock, yBlock, zBlock, d.x, d.y, d.z);

        auto cutSphereOut = [r, rSphere](auto x, auto y, auto z)
        {
            using T_ = decltype(x);
            util::array<T_, 3> X{x, y, z};
            util::array<T_, 3> center{r, r, r};
            return std::sqrt(norm2(X - center)) > rSphere;
        };
        selectParticles(d.x, d.y, d.z, cutSphereOut);

        // create the high-density blob
        std::vector<T> xBlob, yBlob, zBlob;
        cstone::Box<T> boxS(r - rSphere, r + rSphere);
        assembleCube<T>(keyStart, keyEnd, boxS, multiplicity, xBlock, yBlock, zBlock, xBlob, yBlob, zBlob);
        auto keepSphere = [r, rSphere](auto x, auto y, auto z)
        {
            using T_ = decltype(x);
            util::array<T_, 3> X{x, y, z};
            util::array<T_, 3> center{r, r, r};
            return std::sqrt(norm2(X - center)) < rSphere;
        };
        selectParticles(xBlob, yBlob, zBlob, keepSphere);
        std::copy(xBlob.begin(), xBlob.end(), std::back_inserter(d.x));
        std::copy(yBlob.begin(), yBlob.end(), std::back_inserter(d.y));
        std::copy(zBlob.begin(), zBlob.end(), std::back_inserter(d.z));

        // Calculate particle mass with the internal sphere
        // T innerSide   = rSphere;
        // T innerVolume = (4. / 3.) * M_PI * innerSide * innerSide * innerSide;

        size_t numParticlesInternal = blockSize * std::pow(multiplicity, 3);
        // T massPart    = innerVolume * rhoInt / numParticlesInternal;
        T massPart = globalBox.lx() * globalBox.ly() * globalBox.lz() * rhoExt / (2 * numParticlesInternal);

        // Initialize Wind shock domain variables
        d.resize(d.x.size());
        initWindShockFields(d, constants_, massPart);

        return globalBox;
    }

    const std::map<std::string, double>& constants() const override { return constants_; }
};

} // namespace sphexa
