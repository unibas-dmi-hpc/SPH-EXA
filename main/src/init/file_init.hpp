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
 * @brief Simulation data initialization from an HDF5 file
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <map>
#include <random>

#include "cstone/sfc/box.hpp"

#include "io/arg_parser.hpp"
#include "io/factory.hpp"
#include "isim_init.hpp"

namespace sphexa
{

template<class Dataset>
void restoreDataset(IFileReader* reader, int rank, Dataset& d)
{
    d.loadOrStoreAttributes(reader);
    d.resize(reader->localNumParticles());

    auto fieldPointers = d.data();
    for (size_t i = 0; i < fieldPointers.size(); ++i)
    {
        if (d.isConserved(i))
        {
            if (rank == 0) { std::cout << "restoring " << d.fieldNames[i] << std::endl; }
            std::visit([reader, key = d.fieldNames[i]](auto field)
                       { reader->readField(Dataset::prefix + key, field->data()); },
                       fieldPointers[i]);
        }
    }
}

template<class SimulationData>
auto restoreData(IFileReader* reader, int rank, SimulationData& simData)
{
    using T = typename SimulationData::RealType;

    cstone::Box<T> box(0, 1);
    box.loadOrStore(reader);

    restoreDataset(reader, rank, simData.hydro);
    restoreDataset(reader, rank, simData.chem);

    simData.hydro.iteration++;

    return box;
}

template<class Dataset>
class FileInit : public ISimInitializer<Dataset>
{
    mutable std::map<std::string, double> constants_;
    std::string                           h5_fname;

    int initStep = -1;

public:
    FileInit(const std::string& fname)
        : h5_fname(strBeforeSign(fname, ":"))
        , initStep(numberAfterSign(fname, ":"))
    {
    }

    cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, size_t /*n*/, Dataset& simData) const override
    {
        std::unique_ptr<IFileReader> reader;
        reader = std::make_unique<H5PartReader>(simData.comm);
        reader->setStep(h5_fname, initStep);

        auto box = restoreData(reader.get(), rank, simData);

        // Read file attributes and put them in constants_ such that they propagate to the new output after a restart
        auto fileAttributes = reader->fileAttributes();
        for (const auto& attr : fileAttributes)
        {
            int64_t sz = reader->fileAttributeSize(attr);
            if (sz == 1)
            {
                constants_[attr] = 0;
                reader->fileAttribute(attr, &constants_[attr], sz);
            }
        }

        reader->closeStep();

        return box;
    }

    [[nodiscard]] const std::map<std::string, double>& constants() const override { return constants_; }
};

template<class Dataset>
class FileSplitInit : public ISimInitializer<Dataset>
{
    mutable std::map<std::string, double> constants_;
    std::string                           h5_fname;

    int numSplits;

public:
    FileSplitInit(const std::string& fname)
        : h5_fname(strBeforeSign(fname, ","))
        , numSplits(numberAfterSign(fname, ","))
    {
        if (numSplits < 1)
        {
            throw std::runtime_error("Number of particle splits must be a positive integer. Provided value: " +
                                     std::to_string(numSplits));
        }
    }

    cstone::Box<typename Dataset::RealType> init(int rank, int /*nrank*/, size_t /*n*/, Dataset& simData) const override
    {
        std::unique_ptr<IFileReader> reader;
        reader = std::make_unique<H5PartReader>(simData.comm);
        reader->setStep(h5_fname, -1);

        // Read file attributes and put them in constants_ such that they propagate to the new output after a restart
        auto fileAttributes = reader->fileAttributes();
        for (const auto& attr : fileAttributes)
        {
            int64_t sz = reader->fileAttributeSize(attr);
            if (sz == 1)
            {
                constants_[attr] = 0;
                reader->fileAttribute(attr, &constants_[attr], sz);
            }
        }

        size_t numParticlesInFile = reader->localNumParticles();
        size_t numParticlesSplit  = numParticlesInFile * numSplits;

        using T = typename Dataset::RealType;
        cstone::Box<T> box(0, 1);
        box.loadOrStore(reader.get());

        auto& d = simData.hydro;
        d.loadOrStoreAttributes(reader.get());

        d.numParticlesGlobal = reader->globalNumParticles() * numSplits;
        d.iteration          = 1;
        d.ttot               = 0.0;
        d.minDt /= (100 * numSplits);
        d.minDt_m1 /= (100 * numSplits);

        d.x.resize(numParticlesSplit);
        d.y.resize(numParticlesSplit);
        d.z.resize(numParticlesSplit);
        d.h.resize(numParticlesSplit);
        {
            std::vector<T> x0(numParticlesInFile), y0(numParticlesInFile), z0(numParticlesInFile),
                h0(numParticlesInFile);
            reader->readField("x", x0.data());
            reader->readField("y", y0.data());
            reader->readField("z", z0.data());
            reader->readField("h", h0.data());

#pragma omp parallel
            {
                std::mt19937                      eng(rank);
                std::uniform_real_distribution<T> rng(0, 1);

                auto ballPoint = [&rng, &eng]()
                {
                    cstone::Vec3<T> X;
                    do
                    {
                        X = cstone::Vec3<T>{rng(eng), rng(eng), rng(eng)};
                    } while (norm2(X) > T(1));

                    return X;
                };

                T hScale = T(1) / std::cbrt(numSplits);

#pragma omp for schedule(static)
                for (size_t i = 0; i < numParticlesInFile; ++i)
                {
                    size_t sIdx = numSplits * i;
                    T      hi   = h0[i];

                    d.x[sIdx] = x0[i];
                    d.y[sIdx] = y0[i];
                    d.z[sIdx] = z0[i];
                    d.h[sIdx] = hi * hScale;

                    for (size_t j = 1; j < numSplits; ++j)
                    {
                        // a random point within a unit sphere
                        auto displacement = ballPoint();

                        d.x[sIdx + j] = x0[i] + hi * displacement[0];
                        d.y[sIdx + j] = y0[i] + hi * displacement[1];
                        d.z[sIdx + j] = z0[i] + hi * displacement[2];
                        d.h[sIdx + j] = hi * hScale;
                    }
                }
            }
        }

        auto replicateField = [numParticlesInFile, numParticlesSplit, this](IFileReader* reader, const std::string& key,
                                                                            auto& dest, T scale)
        {
            std::vector<T> src(numParticlesInFile);
            reader->readField(key, src.data());
            dest.resize(numParticlesSplit);
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < numParticlesInFile; ++i)
            {
                size_t sIdx = numSplits * i;
                std::fill(dest.data() + sIdx, dest.data() + sIdx + numSplits, src[i] * scale);
            }
        };

        d.resize(numParticlesSplit);
        replicateField(reader.get(), "m", d.m, T(1) / numSplits);
        replicateField(reader.get(), "vx", d.vx, T(1));
        replicateField(reader.get(), "vy", d.vy, T(1));
        replicateField(reader.get(), "vz", d.vz, T(1));
        replicateField(reader.get(), "temp", d.temp, T(1));

        std::fill(d.du_m1.begin(), d.du_m1.end(), 0);
        std::transform(d.vx.begin(), d.vx.end(), d.x_m1.begin(), [dt = d.minDt](auto v_) { return v_ * dt; });
        std::transform(d.vy.begin(), d.vy.end(), d.y_m1.begin(), [dt = d.minDt](auto v_) { return v_ * dt; });
        std::transform(d.vz.begin(), d.vz.end(), d.z_m1.begin(), [dt = d.minDt](auto v_) { return v_ * dt; });

        if (d.isAllocated("alpha"))
        {
            try
            {
                replicateField(reader.get(), "alpha", d.alpha, T(1));
            }
            catch (std::runtime_error&)
            {
                std::fill(d.alpha.begin(), d.alpha.end(), d.alphamin);
            }
        }

        reader->closeStep();

        return box;
    }

    [[nodiscard]] const std::map<std::string, double>& constants() const override { return constants_; }
};

} // namespace sphexa
