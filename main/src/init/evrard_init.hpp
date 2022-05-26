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

#include "isim_init.hpp"
#include "grid.hpp"

namespace sphexa
{

template<class Dataset>
void initEvrardFields(Dataset& d, const std::map<std::string, double>& constants)
{
    using T = typename Dataset::RealType;

    int    ng0           = 100;
    double mPart         = constants.at("mTotal") / d.numParticlesGlobal;
    double firstTimeStep = constants.at("firstTimeStep");

    std::fill(d.m.begin(), d.m.end(), mPart);
    std::fill(d.du_m1.begin(), d.du_m1.end(), 0.0);
    std::fill(d.mui.begin(), d.mui.end(), 10.0);
    std::fill(d.alpha.begin(), d.alpha.end(), d.alphamin);

    d.minDt    = firstTimeStep;
    d.minDt_m1 = firstTimeStep;

    std::fill(d.vx.begin(), d.vx.end(), 0.0);
    std::fill(d.vy.begin(), d.vy.end(), 0.0);
    std::fill(d.vz.begin(), d.vz.end(), 0.0);
    std::fill(d.u.begin(), d.u.end(), constants.at("u0"));

    T totalVolume = 4 * M_PI / 3 * std::pow(constants.at("r"), 3);
    // before the contraction with sqrt(r), the sphere has a constant particle concentration of Ntot / Vtot
    // after shifting particles towards the center by factor sqrt(r), the local concentration becomes
    // c(r) = 2/3 * 1/r * Ntot / Vtot
    T c0 = 2. / 3. * d.numParticlesGlobal / totalVolume;

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < d.x.size(); i++)
    {
        T radius0 = std::sqrt((d.x[i] * d.x[i]) + (d.y[i] * d.y[i]) + (d.z[i] * d.z[i]));

        // multiply coordinates by sqrt(r) to generate a density profile ~ 1/r
        T contraction = std::sqrt(radius0);
        d.x[i] *= contraction;
        d.y[i] *= contraction;
        d.z[i] *= contraction;
        T radius = radius0 * contraction;

        T concentration = c0 / radius;
        d.h[i]          = std::cbrt(3 / (4 * M_PI) * ng0 / concentration) * 0.5;

        d.x_m1[i] = d.x[i] - d.vx[i] * firstTimeStep;
        d.y_m1[i] = d.y[i] - d.vy[i] * firstTimeStep;
        d.z_m1[i] = d.z[i] - d.vz[i] * firstTimeStep;
    }
}

std::map<std::string, double> evrardConstants()
{
    return {{"G", 1.}, {"r", 1.}, {"mTotal", 1.}, {"gamma", 5. / 3.}, {"u0", 0.05}, {"firstTimeStep", 1e-4}};
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

    cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, size_t cbrtNumPart, Dataset& d) const override
    {
        using KeyType = typename Dataset::KeyType;
        using T       = typename Dataset::RealType;

        std::vector<T> xBlock, yBlock, zBlock;
        fileutils::readTemplateBlock(glassBlock, xBlock, yBlock, zBlock);
        size_t blockSize = xBlock.size();

        size_t multiplicity = std::rint(cbrtNumPart / std::cbrt(blockSize));

        d.g = constants_.at("G");
        T r = constants_.at("r");

        cstone::Box<T> globalBox(-r, r, false);

        auto [keyStart, keyEnd] = partitionRange(cstone::nodeRange<KeyType>(0), rank, numRanks);
        assembleCube<T>(keyStart, keyEnd, globalBox, multiplicity, xBlock, yBlock, zBlock, d.x, d.y, d.z);
        cutSphere(r, d.x, d.y, d.z);

        d.numParticlesGlobal = d.x.size();
        MPI_Allreduce(MPI_IN_PLACE, &d.numParticlesGlobal, 1, MpiType<size_t>{}, MPI_SUM, d.comm);

        size_t                    bucketSize = std::max(64lu, d.numParticlesGlobal / (100 * numRanks));
        cstone::BufferDescription bufDesc{0, cstone::LocalIndex(d.x.size()), cstone::LocalIndex(d.x.size())};

        cstone::GlobalAssignment<KeyType, T> distributor(rank, numRanks, bucketSize, globalBox);
        cstone::ReorderFunctor_t<cstone::CpuTag, T, KeyType, cstone::LocalIndex> reorderFunctor;

        std::vector<KeyType> particleKeys(d.x.size());
        cstone::LocalIndex   newNParticlesAssigned =
            distributor.assign(bufDesc, reorderFunctor, particleKeys.data(), d.x.data(), d.y.data(), d.z.data());
        size_t exchangeSize = std::max(d.x.size(), size_t(newNParticlesAssigned));
        cstone::reallocate(exchangeSize, particleKeys, d.x, d.y, d.z);
        auto [exchangeStart, keyView] =
            distributor.distribute(bufDesc, reorderFunctor, particleKeys.data(), d.x.data(), d.y.data(), d.z.data());

        reorderFunctor(d.x.data() + exchangeStart, d.x.data());
        reorderFunctor(d.y.data() + exchangeStart, d.y.data());
        reorderFunctor(d.z.data() + exchangeStart, d.z.data());
        d.x.resize(keyView.size());
        d.y.resize(keyView.size());
        d.z.resize(keyView.size());
        d.x.shrink_to_fit();
        d.y.shrink_to_fit();
        d.z.shrink_to_fit();

        d.resize(d.x.size());
        initEvrardFields(d, constants_);

        return globalBox;
    }

    const std::map<std::string, double>& constants() const override { return constants_; }
};

} // namespace sphexa
