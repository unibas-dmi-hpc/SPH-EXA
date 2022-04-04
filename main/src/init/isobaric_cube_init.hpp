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
 * @brief Isobaric cube simulation data initialization
 *
 * @author Jose A. Escartin <ja.escartin@gmail.com>"
 */

#pragma once

#include <map>
#include <cmath>
#include <algorithm>

#include "cstone/sfc/box.hpp"

#include "isim_init.hpp"
#include "grid.hpp"

namespace sphexa
{

template<class Dataset>
void initIsobaricCubeFields(Dataset& d, const std::map<std::string, double>& constants, double massPart)
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
    T epsilon       = constants.at("epsilon");

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
        if ((abs(d.x[i]) - r > epsilon) || (abs(d.y[i]) - r > epsilon) || (abs(d.z[i]) - r > epsilon))
        {
            d.h[i] = hExt;
            d.u[i] = uExt;
        }
        else
        {
            d.h[i] = hInt;
            d.u[i] = uInt;
        }

        d.vx[i] = 0.;
        d.vy[i] = 0.;
        d.vz[i] = 0.;

        d.x_m1[i] = d.x[i] - d.vx[i] * firstTimeStep;
        d.y_m1[i] = d.y[i] - d.vy[i] * firstTimeStep;
        d.z_m1[i] = d.z[i] - d.vz[i] * firstTimeStep;
    }
}

std::map<std::string, double> IsobaricCubeConstants()
{
    return {{"r", .25},
            {"rDelta", .25},
            {"dim", 3},
            {"gamma", 5.0 / 3.0},
            {"rhoExt", 1.},
            {"rhoInt", 8.},
            {"pIsobaric", 2.5}, // pIsobaric = (gamma − 1.) * rho * u
            {"firstTimeStep", 1e-4},
            {"epsilon", 1e-15},
            {"pairInstability", 1e-3}};
}

template<class Dataset>
class IsobaricCubeGrid : public ISimInitializer<Dataset>
{
    std::map<std::string, double> constants_;

public:
    IsobaricCubeGrid() { constants_ = IsobaricCubeConstants(); }

    cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, size_t cubeSide, Dataset& d) const override
    {
        using T = typename Dataset::RealType;

        size_t nIntPart      = cubeSide * cubeSide * cubeSide;
        d.numParticlesGlobal = nIntPart;
        auto [first, last]   = partitionRange(d.numParticlesGlobal, rank, numRanks);
        resize(d, last - first);

        T r = constants_.at("r");
        regularGrid(r, cubeSide, first, last, d.x, d.y, d.z);

        T stepRatio = constants_.at("rhoInt") / constants_.at("rhoExt");
        T rDelta    = constants_.at("rDelta");
        T stepInt   = (2. * r) / cubeSide;
        T stepExt   = stepInt * std::pow(stepRatio, 1. / 3.);
        T totalSide = 2. * (r + rDelta);

        size_t extCubeSide = round(totalSide / stepExt);
        stepExt            = totalSide / T(extCubeSide); // Adjust stepExt exactly to the actual # of particles

        T initR       = -(r + rDelta) + 0.5 * stepExt;
        T totalVolume = totalSide * totalSide * totalSide;

        size_t totalCubeExt = extCubeSide * extCubeSide * extCubeSide;
        T      massPart     = totalVolume / totalCubeExt;

        // Count additional particles
        size_t nExtPart = 0;

        T epsilon = constants_.at("epsilon");
        for (size_t i = 0; i < extCubeSide; i++)
        {
            T lz = initR + (i * stepExt);

            for (size_t j = 0; j < extCubeSide; j++)
            {
                T ly = initR + (j * stepExt);

                for (size_t k = 0; k < extCubeSide; k++)
                {
                    T lx = initR + (k * stepExt);

                    if ((abs(lx) - r > epsilon) || (abs(ly) - r > epsilon) || (abs(lz) - r > epsilon)) { nExtPart++; }
                }
            }
        }

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
                    if ((abs(lx) - r > epsilon) || (abs(ly) - r > epsilon) || (abs(lz) - r > epsilon))
                    {
                        d.x[idx] = lx;
                        d.y[idx] = ly;
                        d.z[idx] = lz;

                        idx++;
                    }
                }
            }
        }

        initIsobaricCubeFields(d, constants_, massPart);

        return cstone::Box<T>(-(r + rDelta), r + rDelta, true);
    }

    const std::map<std::string, double>& constants() const override { return constants_; }
};

template<class Dataset>
class IsobaricCubeGlass : public ISimInitializer<Dataset>
{
    std::string                   glassBlock;
    std::map<std::string, double> constants_;

public:
    IsobaricCubeGlass(std::string initBlock)
        : glassBlock(initBlock)
    {
        constants_ = IsobaricCubeConstants();
    }

    cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, size_t cbrtNumPart, Dataset& d) const override
    {
        using KeyType = typename Dataset::KeyType;
        using T       = typename Dataset::RealType;

        T r               = constants_.at("r");
        T rDelta          = constants_.at("rDelta");
        T pairInstability = constants_.at("pairInstability");
        T rLimit          = r + pairInstability;

        // Load glass for internal glass cube [0, 1]
        std::vector<T> xBlockInt, yBlockInt, zBlockInt;
        fileutils::readTemplateBlock(glassBlock, xBlockInt, yBlockInt, zBlockInt);
        size_t blockSizeInt = xBlockInt.size();

        // Copy glass for external cube
        std::vector<T> xBlockExt = xBlockInt;
        std::vector<T> yBlockExt = yBlockInt;
        std::vector<T> zBlockExt = zBlockInt;

        // Reduce by half the coordinates in the internal cube -> Density ratio 1 to 8, if the cubes have same #parts
        T ratioInt = .5;
        std::transform(
            xBlockInt.begin(), xBlockInt.end(), xBlockInt.begin(), [ratioInt](T& c) { return ratioInt * c; });
        std::transform(
            yBlockInt.begin(), yBlockInt.end(), yBlockInt.begin(), [ratioInt](T& c) { return ratioInt * c; });
        std::transform(
            zBlockInt.begin(), zBlockInt.end(), zBlockInt.begin(), [ratioInt](T& c) { return ratioInt * c; });

        /*
                for (size_t i = 0; i < blockSizeInt; i++)
                {
                    std::cout << "TotalIntPart=" << blockSizeInt << ", i=" << i << "; x=" << xBlockInt[i] << ",y=" <<
           yBlockInt[i] << ",z=" << zBlockInt[i] << std::endl;
                }
        */
        // Remove the particles in the external cube, that are in the internal cube space
        size_t blockSizeExt = 0;

        typename std::vector<T>::iterator xIt = xBlockExt.begin();
        typename std::vector<T>::iterator yIt = yBlockExt.begin();
        typename std::vector<T>::iterator zIt = zBlockExt.begin();

        while (xIt != xBlockExt.end())
        {
            // std::cout << "i=" << blockSizeExt << "; x=" << *xIt << ",y=" << *yIt << ",z=" << *zIt << std::endl;

            if ((abs(*xIt) < rLimit) && (abs(*yIt) < rLimit) && (abs(*zIt) < rLimit))
            {
                // std::cout << "Remove particle=" << blockSizeExt << "; x=" << *xIt << ",y=" << *yIt << ",z=" << *zIt
                // << std::endl;

                xIt = xBlockExt.erase(xIt);
                yIt = yBlockExt.erase(yIt);
                zIt = zBlockExt.erase(zIt);
            }
            else
            {
                // std::cout << "Add    particle=" << blockSizeExt << "; x=" << *xIt << ",y=" << *yIt << ",z=" << *zIt
                // << std::endl;

                ++xIt;
                ++yIt;
                ++zIt;

                ++blockSizeExt;
            }
        }
        /*
                for (size_t i = 0; i < blockSizeInt; i++)
                {
                    std::cout << "TotalExtPart=" << blockSizeExt << ", i=" << i << "; x=" << xBlockExt[i] << ",y=" <<
           yBlockExt[i] << ",z=" << zBlockExt[i] << std::endl;
                }
        */
        // Concatenate internal + external cube
        size_t blockSize = blockSizeInt + blockSizeExt;

        // Concatenate x-positions
        std::vector<T> xBlock;
        xBlock.reserve(blockSize);
        xBlock.insert(xBlock.end(), xBlockInt.begin(), xBlockInt.end());
        xBlock.insert(xBlock.end(), xBlockExt.begin(), xBlockExt.end());

        // Concatenate y-positions
        std::vector<T> yBlock;
        yBlock.reserve(blockSize);
        yBlock.insert(yBlock.end(), yBlockInt.begin(), yBlockInt.end());
        yBlock.insert(yBlock.end(), yBlockExt.begin(), yBlockExt.end());

        // Concatenate z-positions
        std::vector<T> zBlock;
        zBlock.reserve(blockSize);
        zBlock.insert(zBlock.end(), zBlockInt.begin(), zBlockInt.end());
        zBlock.insert(zBlock.end(), zBlockExt.begin(), zBlockExt.end());

        for (size_t i = 0; i < blockSize; i++)
        {
            std::cout << "TotalPart=" << blockSize << ", i=" << i << "; x=" << xBlock[i] << ",y=" << yBlock[i]
                      << ",z=" << zBlock[i] << std::endl;
        }

        // Make Box and resize domine
        size_t multiplicity  = std::rint(cbrtNumPart / std::cbrt(blockSize));
        d.numParticlesGlobal = multiplicity * multiplicity * multiplicity * blockSize;

        cstone::Box<T> globalBox(-(r + rDelta), r + rDelta, true);

        auto [keyStart, keyEnd] = partitionRange(cstone::nodeRange<KeyType>(0), rank, numRanks);
        assembleCube<T>(keyStart, keyEnd, globalBox, multiplicity, xBlock, yBlock, zBlock, d.x, d.y, d.z);
        resize(d, d.x.size());

        // std::cout << "Assembled cube!" << std::endl;

        // Calculate mass particle
        T totalSide   = 2. * (r + rDelta);
        T totalVolume = totalSide * totalSide * totalSide;
        T massPart    = totalVolume / d.x.size();

        // Initialize Isobaric cube domine variables
        initIsobaricCubeFields(d, constants_, massPart);

        // std::cout << "Cube box created!" << std::endl;

        return globalBox;
    }

    const std::map<std::string, double>& constants() const override { return constants_; }
};

} // namespace sphexa
