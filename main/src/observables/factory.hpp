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

namespace sphexa
{

#ifdef SPH_EXA_HAVE_H5PART

//! @brief return true if the specified attribute exists, is integral and is non-zero and if @p fname exists
static bool haveH5Attribute(const std::string& fname, const std::string& attributeToRead)
{
    int ret = 0;

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
            if (attributeType != H5PART_INT64)
            {
                throw std::runtime_error("unexpected attribute type in haveH5Attribute\n");
            }

            if (attributeName == attributeToRead)
            {
                H5PartReadFileAttrib(h5_file, attributeName, &ret);
                break;
            }
        }
        H5PartCloseFile(h5_file);
    }

    return ret;
}

#else

static bool haveH5Attribute(const std::string& fname, const std::string& attributeToRead)
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
    if (haveH5Attribute(testCase, "KelvinHelmholtzGrowthRate"))
    {
        return std::make_unique<TimeEnergyGrowth<Dataset>>(constantsFile);
    }
    else
    {
        return std::make_unique<TimeAndEnergy<Dataset>>(constantsFile);
    }
}

} // namespace sphexa
