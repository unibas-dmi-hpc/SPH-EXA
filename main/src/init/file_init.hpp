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

#include "cstone/sfc/box.hpp"

#include "io/factory.hpp"
#include "isim_init.hpp"

namespace sphexa
{

template<class HydroData>
cstone::Box<typename HydroData::RealType> restoreHydroData(IFileReader* reader, int rank, HydroData& d)
{
    using T = typename HydroData::RealType;

    cstone::Box<T> box(0, 1);
    box.loadOrStore(reader);

    d.loadOrStoreAttributes(reader);
    d.iteration++;
    d.resize(reader->localNumParticles());

    if (d.numParticlesGlobal != reader->globalNumParticles())
    {
        throw std::runtime_error("numParticlesGlobal mismatch\n");
    }

    auto fieldPointers = d.data();
    for (size_t i = 0; i < fieldPointers.size(); ++i)
    {
        if (d.isConserved(i))
        {
            if (rank == 0) { std::cout << "restoring " << d.fieldNames[i] << std::endl; }
            std::visit([reader, key = d.fieldNames[i]](auto field) { reader->readField(key, field->data()); },
                       fieldPointers[i]);
        }
    }

    return box;
}

template<class Dataset>
class FileInit : public ISimInitializer<Dataset>
{
    std::map<std::string, double> constants_;
    std::string                   h5_fname;

public:
    FileInit(std::string fname)
        : h5_fname(fname)
    {
    }

    cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, size_t /*n*/, Dataset& simData) const override
    {
        std::unique_ptr<IFileReader> reader;
        reader = std::make_unique<H5PartReader>(simData.comm);
        reader->setStep(h5_fname, -1);

        auto box = restoreHydroData(reader.get(), rank, simData.hydro);

        reader->closeStep();

        return box;
    }

    const std::map<std::string, double>& constants() const override { return constants_; }
};

} // namespace sphexa
