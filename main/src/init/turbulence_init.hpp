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

#include "isim_init.hpp"
#include "grid.hpp"

#include "st_ounoise.hpp"
#include "stir_init.hpp"
namespace sphexa
{

template<class Dataset>
void initTurbulenceFields(Dataset& d, const std::map<std::string, double>& constants)
{
    using T = typename Dataset::RealType;

    T firstTimeStep       = constants.at("firstTimeStep");
    T eps                 = constants.at("epsilon");
    size_t stMaxModes     = constants.at("stMaxModes");
    T Lbox                = constants.at("Lbox");
    T stMachVelocity      = constants.at("stMachVelocity");
    size_t ndim           = constants.at("ndim");
    size_t stSpectForm    = constants.at("stSpectForm");

    T twopi=8.0*std::atan(1.0);
    T stEnergy = 5.0e-3 * pow(stMachVelocity,3)/Lbox;
    d.stDecay = Lbox/(2.0*velocity);
    T  stStirMin = (1.e0-eps) * twopi/Lbox;
    T stStirMax = (3.e0+eps) * twopi/Lbox;
    d.stSeed=stSeedIni;

    stir_init(Lbox,Lbox,Lbox,stMaxModes,d.stOUvar,stEnergy,d.stDecay,stStirMax,stStirMin,
              ndim,d.stSolWeightNorm,d.stSolweight,d.stNModes,d.stAmpl,d.stMode,stSpectForm);
    st_ounoiseinit(d.stOUphases, 6*d.stNModes, d.stOUvar, d.stSeed);

    std::fill(d.m.begin(), d.m.end(), massPart);
    std::fill(d.du_m1.begin(), d.du_m1.end(), 0.0);
    std::fill(d.dt.begin(), d.dt.end(), firstTimeStep);
    std::fill(d.dt_m1.begin(), d.dt_m1.end(), firstTimeStep);
    std::fill(d.alpha.begin(), d.alpha.end(), d.alphamin);

    d.minDt    = firstTimeStep;
    d.minDt_m1 = firstTimeStep;

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < d.x.size(); i++)
    {

        d.vx[i] = 0.;
        d.vy[i] = 0.;
        d.vz[i] = 0.;

        d.x_m1[i] = d.x[i] - d.vx[i] * firstTimeStep;
        d.y_m1[i] = d.y[i] - d.vy[i] * firstTimeStep;
        d.z_m1[i] = d.z[i] - d.vz[i] * firstTimeStep;
    }
}

std::map<std::string, double> TurbulenceConstants()
{
    return {
            {"stMaxModes", 100000}
            {"Lbox", 1.0},
            {"stMachVelocity", 0.3e0},
            {"dim", 3},
            {"firstTimeStep", 1e-4},
            {"epsilon", 1e-15};
            {"stSeedIni", 251299};
            {"stSpectForm", 1};
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

        cstone::Box<T> globalBox(-2 * r, 2 * r, true);
        auto [keyStart, keyEnd] = partitionRange(cstone::nodeRange<KeyType>(0), rank, numRanks);
        assembleCube<T>(keyStart, keyEnd, globalBox, multiplicity, xBlock, yBlock, zBlock, d.x, d.y, d.z);  //

        // Initialize isobaric cube domain variables
        resize(d, d.x.size());
        initTurbulenceFields(d, constants_);

        return globalBox;
    }

    const std::map<std::string, double>& constants() const override { return constants_; }
};

} // namespace sphexa
