/*
 * MIT License
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *               2024 University of Basel
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
 * @brief Data initialization for the SPHERIC validation test #16
 *
 * @author Lukas Schmidt
 */

#pragma once

#include <map>

#include "cstone/sfc/box.hpp"

#include "isim_init.hpp"
#include "early_sync.hpp"
#include "grid.hpp"

namespace sphexa
{

InitSettings sphericVortexConstants()
{
    return {{"L", 1.}, {"p0", 2.5}, {"rho0", 1.}, {"U0", 1.}, {"minDt", 1e-7}, {"minDt_m1", 1e-7}};
}

template<class Dataset>
void initVortexFields(Dataset& d, const std::map<std::string, double>& constants)
{
    using T = typename Dataset::RealType;

    double L           = constants.at("L");
    double totalVolume = std::pow(L, 3);
    double hInit       = std::cbrt(3.0 / (4 * M_PI) * d.ng0 * totalVolume / d.numParticlesGlobal) * 0.5;
    double k           = 2 * M_PI / L;
    double p0          = constants.at("p0");
    double rho0        = constants.at("rho0");

    double mPart = constants.at("rho0") * totalVolume / d.numParticlesGlobal;

    std::fill(d.m.begin(), d.m.end(), mPart);
    std::fill(d.h.begin(), d.h.end(), hInit);
    std::fill(d.du_m1.begin(), d.du_m1.end(), 0.0);
    std::fill(d.mui.begin(), d.mui.end(), d.muiConst);
    std::fill(d.alpha.begin(), d.alpha.end(), d.alphamin);

    // general form: d.x_m1[i] = d.vx[i] * firstTimeStep;
    std::fill(d.x_m1.begin(), d.x_m1.end(), 0.0);
    std::fill(d.y_m1.begin(), d.y_m1.end(), 0.0);
    std::fill(d.z_m1.begin(), d.z_m1.end(), 0.0);

    auto cv = sph::idealGasCv(d.muiConst, d.gamma);

    // If temperature is not allocated, we can still use this initializer for just the coordinates
    if (d.temp.empty()) { return; }

    T A      = 4 * sqrt(2) / (3 * sqrt(3)) * constants.at("U0");
    T M_PI_3 = M_PI / 3.;

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < d.x.size(); i++)
    {
        T kx = d.x[i] * k;
        T ky = d.y[i] * k;
        T kz = d.z[i] * k;

        //Initial velocities from setting t=0 in the analytical velocity field
        d.vx[i] = A * (sin(kx - M_PI_3) * cos(ky + M_PI_3) * sin(kz + M_PI_2) -
                       cos(kz - M_PI_3) * sin(kx + M_PI_3) * sin(ky + M_PI_2));

        d.vy[i] = A * (sin(ky - M_PI_3) * cos(kz + M_PI_3) * sin(kx + M_PI_2) -
                       cos(kx - M_PI_3) * sin(ky + M_PI_3) * sin(kz + M_PI_2));

        d.vz[i] = A * (sin(kz - M_PI_3) * cos(kx + M_PI_3) * sin(ky + M_PI_2) -
                       cos(ky - M_PI_3) * sin(kz + M_PI_3) * sin(kx + M_PI_2));

        T normVi = d.vx[i] * d.vx[i] + d.vy[i] * d.vy[i] + d.vz[i] * d.vz[i];
        T pi    = p0 - rho0 * normVi / 2;

        d.temp[i] = pi / ((d.gamma - 1.) * rho0) / cv;

        d.x_m1[i] = d.vx[i] * d.minDt;
        d.y_m1[i] = d.vy[i] * d.minDt;
        d.z_m1[i] = d.vz[i] * d.minDt;
    }
}

template<class Dataset>
class SphericVortexGlass : public ISimInitializer<Dataset>
{
    std::string          glassBlock;
    mutable InitSettings settings_;

public:
    SphericVortexGlass(std::string initBlock, std::string settingsFile, IFileReader* reader)
        : glassBlock(std::move(initBlock))
    {
        Dataset d;
        settings_ = buildSettings(d, sphericVortexConstants(), settingsFile, reader);
    }

    /*! @brief initialize particle data with a constant density cube
     *
     * @param[in]    rank             MPI rank ID
     * @param[in]    numRanks         number of MPI ranks
     * @param[in]    cbrtNumPart      the cubic root of the global number of particles to generate
     * @param[inout] d                particle dataset
     * @return                        the global coordinate bounding box
     */
    cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, size_t cbrtNumPart, Dataset& simData,
                                                 IFileReader* reader) const override
    {
        auto& d       = simData.hydro;
        using KeyType = typename Dataset::KeyType;
        using T       = typename Dataset::RealType;

        std::vector<T> xBlock, yBlock, zBlock;
        readTemplateBlock(glassBlock, reader, xBlock, yBlock, zBlock);
        size_t blockSize = xBlock.size();

        int               multi1D            = std::rint(cbrtNumPart / std::cbrt(blockSize));
        cstone::Vec3<int> multiplicity       = {multi1D, multi1D, multi1D};
        size_t            numParticlesGlobal = multi1D * multi1D * multi1D * blockSize;

        T              L = settings_.at("L");
        cstone::Box<T> globalBox(0, L, cstone::BoundaryType::periodic);

        auto [keyStart, keyEnd] = equiDistantSfcSegments<KeyType>(rank, numRanks, 100);
        assembleCuboid<T>(keyStart, keyEnd, globalBox, multiplicity, xBlock, yBlock, zBlock, d.x, d.y, d.z);

        d.resize(d.x.size());

        settings_["numParticlesGlobal"] = double(numParticlesGlobal);
        BuiltinWriter attributeSetter(settings_);
        d.loadOrStoreAttributes(&attributeSetter);

        initVortexFields(d, settings_);

        return globalBox;
    }

    const InitSettings& constants() const override { return settings_; }
};

} // namespace sphexa
