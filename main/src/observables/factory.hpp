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
 */

#pragma once

#include <string>

#include "cstone/sfc/box.hpp"
#include "iobservables.hpp"
#include "time_energy_growth.hpp"
#include "time_energies.hpp"
#include "gravitational_waves.hpp"
#include "wind_bubble_fraction.hpp"
#include "turbulence_mach_rms.hpp"

namespace sphexa
{

#ifdef SPH_EXA_HAVE_H5PART

//! @brief reads a specified attribute if exists and has the specified type
template<class AttrType>
void findH5Attribute(const std::string& fname, const std::string& attributeToRead, AttrType* attribute,
                           h5part_int64_t h5Type)
{

    if (std::filesystem::exists(fname))
    {
        H5PartFile* h5_file  = nullptr;
        h5_file              = H5PartOpenFile(fname.c_str(), H5PART_READ);
        size_t numAttributes = H5PartGetNumFileAttribs(h5_file);

        h5part_int64_t maxlen = 256;
        char           attributeName[maxlen];

        h5part_int64_t attributeType;
        h5part_int64_t attributeLength;

        for (h5part_int64_t i = 0; i < numAttributes; i++)
        {
            H5PartGetFileAttribInfo(h5_file, i, attributeName, maxlen, &attributeType, &attributeLength);
            if (attributeName == attributeToRead && attributeType == h5Type)
            {

                H5PartReadFileAttrib(h5_file, attributeName, attribute);
                break;
            }
        }
        H5PartCloseFile(h5_file);
    }
}

#else

[[maybe_unused]] static bool haveH5Attribute(const std::string& fname, const std::string& attributeToRead)
{
    if (std::filesystem::exists(fname))
    {
        std::cout << "WARNING: Could not read attribute " + attributeToRead + ". HDF5 support missing\n";
    }

    return false;
}

#endif

template<class Dataset>
std::unique_ptr<IObservables<Dataset>> observablesFactory(const std::string& testCase, std::ofstream& constantsFile)
{
#ifdef SPH_EXA_HAVE_H5PART

    std::string    khGrowthRate = "KelvinHelmholtzGrowthRate";
    h5part_int64_t khAttribute;
    findH5Attribute<h5part_int64_t>(testCase, khGrowthRate, &khAttribute, H5PART_INT64);
    if (khAttribute != 0 || testCase == "kelvin-helmholtz")
    {
        return std::make_unique<TimeEnergyGrowth<Dataset>>(constantsFile);
    }

    std::string gravWaves = "observeGravWaves";
    double      gravWaveAttribute[3];
    findH5Attribute<h5part_float64_t>(testCase, gravWaves, gravWaveAttribute, H5PART_FLOAT64);
    if (gravWaveAttribute[0] != 0.0)
    {
        return std::make_unique<GravWaves<Dataset>>(constantsFile, gravWaveAttribute[1], gravWaveAttribute[2]);
    }

    if (testCase == "wind-shock")
    {
        double rhoInt       = WindShockConstants().at("rhoInt");
        double uExt         = WindShockConstants().at("uExt");
        double bubbleVolume = std::pow(WindShockConstants().at("rSphere"), 3) * 4.0 / 3.0 * M_PI;
        double bubbleMass   = bubbleVolume * rhoInt;
        return std::make_unique<WindBubble<Dataset>>(constantsFile, rhoInt, uExt, bubbleMass);
    }
#endif

    if (testCase == "turbulence") { return std::make_unique<TurbulenceMachRMS<Dataset>>(constantsFile); }
    if (testCase == "kelvin-helmholtz") { return std::make_unique<TimeEnergyGrowth<Dataset>>(constantsFile); }

    return std::make_unique<TimeAndEnergy<Dataset>>(constantsFile);
}

} // namespace sphexa
