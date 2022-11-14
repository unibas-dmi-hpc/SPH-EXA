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
 * @author Jose A. Escartin <ja.escartin@gmail.com>
 */

#pragma once

#include <map>
#include <cmath>
#include <algorithm>

#include "cstone/sfc/box.hpp"
#include "sph/eos.hpp"

#include "io/file_utils.hpp"
#include "isim_init.hpp"
#include "grid.hpp"

namespace sphexa
{

std::map<std::string, double> IsobaricCubeConstants()
{
    return {{"r", .25},         {"rDelta", .25},         {"dim", 3},         {"gamma", 5.0 / 3.0},
            {"rhoExt", 1.},     {"rhoInt", 8.},          {"pIsobaric", 2.5}, {"firstTimeStep", 1e-4},
            {"epsilon", 1e-15}, {"pairInstability", 0.}, {"mui", 10.0}};
}

template<class Dataset>
void initIsobaricCubeFields(Dataset& d, const std::map<std::string, double>& constants, double massPart)
{
    using T = typename Dataset::RealType;

    T r      = constants.at("r");
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

    d.gamma    = constants.at("gamma");
    d.muiConst = constants.at("mui");
    d.minDt    = firstTimeStep;
    d.minDt_m1 = firstTimeStep;

    auto cv = sph::idealGasCv(d.muiConst);

    std::fill(d.m.begin(), d.m.end(), massPart);
    std::fill(d.du_m1.begin(), d.du_m1.end(), 0.0);
    std::fill(d.mui.begin(), d.mui.end(), d.muiConst);
    std::fill(d.alpha.begin(), d.alpha.end(), d.alphamin);
    std::fill(d.vx.begin(), d.vx.end(), 0.0);
    std::fill(d.vy.begin(), d.vy.end(), 0.0);
    std::fill(d.vz.begin(), d.vz.end(), 0.0);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < d.x.size(); i++)
    {
        T xi = std::abs(d.x[i]);
        T yi = std::abs(d.y[i]);
        T zi = std::abs(d.z[i]);

        if ((xi > r + epsilon) || (yi > r + epsilon) || (zi > r + epsilon))
        {
            if ((xi > r + 2 * hExt) || (yi > r + 2 * hExt) || (zi > r + 2 * hExt))
            {
                // more than two smoothing lengths away from the inner cube
                d.h[i] = hExt;
            }
            else
            {
                T dist = std::max({xi - r, yi - r, zi - r});
                // reduce smoothing lengths for particles outside, but close to the inner cube
                d.h[i] = hInt * (1 - dist / (2 * hExt)) + hExt * dist / (2 * hExt);
            }

            d.temp[i] = uExt / cv;
        }
        else
        {
            d.h[i]    = hInt;
            d.temp[i] = uInt / cv;
        }

        d.x_m1[i] = d.vx[i] * firstTimeStep;
        d.y_m1[i] = d.vy[i] * firstTimeStep;
        d.z_m1[i] = d.vz[i] * firstTimeStep;
    }
}

/*! @brief compute the shift factor towards the center for point X in a capped pyramid
 *
 * @tparam T      float or double
 * @param  X      a 3D point with at least one coordinate > s and all coordinates < rExt
 * @param  rInt   half cube length of the internal high-density cube
 * @param  s      compression radius used to create the high-density cube, in [rInt, rExt]
 * @param  rExt   half cube length of the external low-density cube
 * @return        factor in [0:1]
 */
template<class T>
T cappedPyramidStretch(cstone::Vec3<T> X, T rInt, T s, T rExt)
{
    assert(rInt < s && s < rExt);

    X = abs(X);

    //! the intersection of the ray from the coordinate origin through X with the outer cube
    cstone::Vec3<T> pointA = X * (rExt / util::max(X));
    //! the intersection of the ray from the coordinate origin through X with the stretch cube [-s, s]^3
    cstone::Vec3<T> pointB = X * (s / util::max(X));
    //! the intersection of the ray from the coordinate origin through X with the inner cube
    cstone::Vec3<T> pointC = X * (rInt / util::max(X));

    // distances of points A, B and C from the coordinate origin
    T hp     = std::sqrt(norm2(pointC));
    T sp     = std::sqrt(norm2(pointB));
    T rp     = std::sqrt(norm2(pointA));
    T radius = std::sqrt(norm2(X));

    /*! transformation map: particle X is moved towards the coordinate origin
     * known mapped values:
     * (1) if X == pointA, X is not moved
     * (2) if X == pointB, X is moved to point C
     *
     * The map is not linear to compensate for the shrinking area of the capped pyramid top and keep density constant.
     */
    T expo = 0.75;
    //! normalization constant to satisfy (1) and (2)
    T a         = (rp - hp) / std::pow(rp - sp, expo);
    T newRadius = a * std::pow(radius - sp, expo) + hp;

    T scaleFactor = newRadius / radius;

    return scaleFactor;
}

/*! returns a value in [rInt:rExt]
 *
 * @tparam T         float or double
 * @param  rInt      inner cube half side
 * @param  rExt      outer dube half side
 * @param  rhoRatio  the desired density ratio between inner and outer
 * @return           value s, such that if [-s, s]^3 gets contracted into the inner cube
 *                   and [s:rExt, s:rExt]^3 is expanded into the resulting empty area,
 *                   the inner and outer cubes will have a density ratio of @p rhoRatio
 *
 * Derivation:
 *      internal density: rho_int = rho_0 * (s / rInt)^3
 *
 *      external density: rho_ext = rho_0  * (2rExt)^3 - (2s)^3
 *                                           ------------------
 *                                           (2rExt)^3 - (2rInt)^3
 *
 * The return value is the solution of rho_int / rho_ext == rhoRatio for s
 */
template<class T>
T computeStretchFactor(T rInt, T rExt, T rhoRatio)
{
    T hc = rInt * rInt * rInt;
    T rc = rExt * rExt * rExt;
    T s  = std::cbrt(rhoRatio * hc * rc / (rc - hc + rhoRatio * hc));
    assert(rInt < s && s < rExt);
    return s;
}

template<class T>
void compressCenterCube(gsl::span<T> x, gsl::span<T> y, gsl::span<T> z, T rInt, T s, T rExt, T epsilon)
{
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < x.size(); i++)
    {
        if ((std::abs(x[i]) - s > epsilon) || (std::abs(y[i]) - s > epsilon) || (std::abs(z[i]) - s > epsilon))
        {
            cstone::Vec3<T> X{x[i], y[i], z[i]};

            T scaleFactor = cappedPyramidStretch(X, rInt, s, rExt);

            x[i] *= scaleFactor;
            y[i] *= scaleFactor;
            z[i] *= scaleFactor;
        }
        else
        {
            // particle in inner high-density region get contracted towards the center
            x[i] *= rInt / s;
            y[i] *= rInt / s;
            z[i] *= rInt / s;
        }
    }
}

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

    cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, size_t cbrtNumPart,
                                                 Dataset& simData) const override
    {
        auto& d       = simData.hydro;
        using KeyType = typename Dataset::KeyType;
        using T       = typename Dataset::RealType;

        T r       = constants_.at("r");
        T rhoInt  = constants_.at("rhoInt");
        T rhoExt  = constants_.at("rhoExt");
        T epsilon = constants_.at("pairInstability");

        std::vector<T> xBlock, yBlock, zBlock;
        fileutils::readTemplateBlock(glassBlock, xBlock, yBlock, zBlock);
        size_t blockSize = xBlock.size();

        size_t multiplicity  = std::rint(cbrtNumPart / std::cbrt(blockSize));
        d.numParticlesGlobal = multiplicity * multiplicity * multiplicity * blockSize;

        cstone::Box<T> globalBox(-2 * r, 2 * r, cstone::BoundaryType::periodic);
        auto [keyStart, keyEnd] = partitionRange(cstone::nodeRange<KeyType>(0), rank, numRanks);
        assembleCube<T>(keyStart, keyEnd, globalBox, multiplicity, xBlock, yBlock, zBlock, d.x, d.y, d.z);

        T s = computeStretchFactor(r, 2 * r, rhoInt / rhoExt);
        compressCenterCube<T>(d.x, d.y, d.z, r, s, 2. * r, epsilon);

        size_t numParticlesInternal = d.numParticlesGlobal * std::pow(s / (2. * r), 3);

        // Calculate particle mass with the internal cube
        T innerSide   = 2. * r;
        T innerVolume = innerSide * innerSide * innerSide;
        T massPart    = innerVolume * rhoInt / numParticlesInternal;

        // Initialize isobaric cube domain variables
        d.resize(d.x.size());
        initIsobaricCubeFields(d, constants_, massPart);

        return globalBox;
    }

    const std::map<std::string, double>& constants() const override { return constants_; }
};

} // namespace sphexa
