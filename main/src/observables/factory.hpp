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
 * @brief Select/calculate data to be printed to constants.txt each iteration
 *
 * @author Lukas Schmidt
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <span>
#include <string>

#include "cstone/sfc/box.hpp"
#include "io/ifile_io.hpp"

#ifdef SPH_EXA_HAVE_H5PART
#include "io/ifile_io_hdf5.hpp"
#endif

#include "iobservables.hpp"
#include "time_energy_growth.hpp"
#include "time_energies.hpp"
#include "gravitational_waves.hpp"
#include "wind_bubble_fraction.hpp"
#include "turbulence_mach_rms.hpp"

namespace sphexa
{

static bool haveAttribute(IFileReader* reader, const std::string& attributeToRead)
{
    if (reader)
    {
        auto attributes = reader->fileAttributes();
        return std::count(attributes.begin(), attributes.end(), attributeToRead);
    }

    return false;
}

template<class Dataset>
std::unique_ptr<IObservables<Dataset>> observablesFactory(const std::string& testCase, std::ofstream& constantsFile)
{
    std::unique_ptr<IFileReader> reader;
    std::string                  testCaseStripped = strBeforeSign(testCase, ",");

#ifdef SPH_EXA_HAVE_H5PART
    if (std::filesystem::exists(testCaseStripped))
    {
        reader = std::make_unique<H5PartReader>(MPI_COMM_WORLD);
        reader->setStep(testCaseStripped, -1);
    }
#endif

    std::string gravWaves = "observeGravWaves";
    if (haveAttribute(reader.get(), gravWaves))
    {
        double gravWaveThetaPhi[2] = {0.0, 0.0};
        reader->fileAttribute(gravWaves, gravWaveThetaPhi, 2);
        return std::make_unique<GravWaves<Dataset>>(constantsFile, gravWaveThetaPhi[0], gravWaveThetaPhi[1]);
    }

#ifdef SPH_EXA_HAVE_H5PART
    if (testCase == "wind-shock" || haveAttribute(reader.get(), "wind-shock"))
    {
        double rhoInt       = WindShockConstants().at("rhoInt");
        double uExt         = WindShockConstants().at("uExt");
        double bubbleVolume = std::pow(WindShockConstants().at("rSphere"), 3) * 4.0 / 3.0 * M_PI;
        double bubbleMass   = bubbleVolume * rhoInt;
        return std::make_unique<WindBubble<Dataset>>(constantsFile, rhoInt, uExt, bubbleMass);
    }
#endif

    if (testCase == "turbulence" || haveAttribute(reader.get(), "turbulence"))
    {
        return std::make_unique<TurbulenceMachRMS<Dataset>>(constantsFile);
    }
    if (testCase == "kelvin-helmholtz" || haveAttribute(reader.get(), "kelvin-helmholtz"))
    {
        return std::make_unique<TimeEnergyGrowth<Dataset>>(constantsFile);
    }

    if (reader) { reader->closeStep(); }
    return std::make_unique<TimeAndEnergy<Dataset>>(constantsFile);
}

} // namespace sphexa
