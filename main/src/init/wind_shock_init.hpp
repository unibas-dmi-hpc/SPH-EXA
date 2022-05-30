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
    T gamma         = constants.at("gamma");
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

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < d.x.size(); i++)
    {
        T xi = std::abs(d.x[i]);
        T yi = std::abs(d.y[i]);
        T zi = std::abs(d.z[i]);

        T rPos = std::sqrt((xi * xi) + (yi * yi) + (zi * zi));

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
            d.h[i] = hInt;
            d.u[i] = uInt;
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



/*! @brief compute the shift factor towards the center for point X in a capped pyramid
 *
 * @tparam T      float or double
 * @param  rPos   point radius < rExt
 * @param  rInt   half cube length of the internal high-density cube
 * @param  s      compression radius used to create the high-density cube, in [rInt, rExt]
 * @param  rExt   half cube length of the external low-density cube
 * @return        factor in [0:1]
 */
template<class T>
T sphereStretch(T rPos, T rInt, T s, T rExt)
{
    assert(rInt < s && s < rExt && rPos <= rExt);

    T expo      = 0.71;
    T ratio     = (rExt - rInt) / std::pow(rExt - s, expo);
    T newRadius = rInt + std::pow(rPos - s, expo) * ratio;

    return newRadius / rPos;
}

/*! returns a value in [rInt:rExt]
 *
 * @tparam T         float or double
 * @param  rInt      inner sphere half side
 * @param  rExt      outer sphere half side
 * @param  rhoRatio  the desired density ratio between inner and outer
 * @return           value s, such that if [-s, s]^3 gets contracted into the inner sphere
 *                   and [s:rExt, s:rExt]^3 is expanded into the resulting empty area,
 *                   the inner and outer spheres will have a density ratio of @p rhoRatio
 *
 * Derivation:
 *      internal density: rho_int = rho_0 * (s / rInt)^3
 *
 *      external density: rho_ext = rho_0  * (2rExt)^3 - (4/3*pi)*(s)^3
 *                                           -----------------------------
 *                                           (2rExt)^3 - (4/3*pi)*(rInt)^3
 *
 * The return value is the solution of rho_int / rho_ext == rhoRatio for s
 */
template<class T>
T WindShockcomputeStretchFactor(T rInt, T rExt, T rhoRatio)
{
    T factor = (4. / 3. ) * M_PI;
    T hc = factor * rInt * rInt * rInt;
    T rc = 8. * rExt * rExt * rExt;
    T s  = std::cbrt(rhoRatio * (hc / factor) * rc / (rc - hc + rhoRatio * hc));
    assert(rInt < s && s < rExt);
    return s;
}

template<class T>
size_t WindShockcompressCenterSphere(gsl::span<T> x, gsl::span<T> y, gsl::span<T> z, T rInt, T s, T rExt, T epsilon)
{
    size_t sum = 0;

#pragma omp parallel for reduction(+: sum) schedule(static)
    for (size_t i = 0; i < x.size(); i++)
    {
        T rPos = std::sqrt((x[i] * x[i]) + (y[i] * y[i]) + (z[i] * z[i]));

        if (rPos - s > epsilon)
        {
            // Only streech particles inside the rExt radius sphere
            if (rPos <= rExt)
            {
                T scaleFactor = sphereStretch(rPos, rInt, s, rExt);

                x[i] *= scaleFactor;
                y[i] *= scaleFactor;
                z[i] *= scaleFactor;
            }
        }
        else
        {
            // particle in inner high-density region get contracted towards the center
            x[i] *= rInt / s;
            y[i] *= rInt / s;
            z[i] *= rInt / s;

            sum++;
        }
    }

    std::cout << "rExt=" << rExt << ", s=" << s << ", rInt=" << rInt << "; Particles in High density region: " << sum << std::endl;

    return sum;
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

        cstone::Box<T> globalBox(-r, r, true);
        auto [keyStart, keyEnd] = partitionRange(cstone::nodeRange<KeyType>(0), rank, numRanks);
        assembleCube<T>(keyStart, keyEnd, globalBox, multiplicity, xBlock, yBlock, zBlock, d.x, d.y, d.z);

        T      s                    = WindShockcomputeStretchFactor(rSphere, r, rhoInt / rhoExt);
        size_t numParticlesInternal = WindShockcompressCenterSphere<T>(d.x, d.y, d.z, rSphere, s, r, epsilon);

        // Calculate particle mass with the internal sphere
        T innerSide   = rSphere;
        T innerVolume = (4. / 3.) * M_PI * innerSide * innerSide * innerSide;
        T massPart    = innerVolume * rhoInt / numParticlesInternal;

        // Initialize Wind shock domain variables
        d.resize(d.x.size());
        initWindShockFields(d, constants_, massPart);

        return globalBox;
    }

    const std::map<std::string, double>& constants() const override { return constants_; }
};

} // namespace sphexa
