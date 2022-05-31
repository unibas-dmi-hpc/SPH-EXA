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
 * @brief Select a simulation initializer based on user input choice
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <filesystem>
#include <map>

#include "cstone/sfc/box.hpp"

#include "isim_init.hpp"
#include "file_init.hpp"
#include "evrard_init.hpp"
#include "sedov_init.hpp"
#include "noh_init.hpp"
#include "isobaric_cube_init.hpp"
#include "wind_shock_init.hpp"

namespace sphexa
{

template<class Dataset>
std::unique_ptr<ISimInitializer<Dataset>> initializerFactory(std::string testCase, std::string glassBlock)
{
    if (testCase == "sedov")
    {
        if (glassBlock.empty()) { return std::make_unique<SedovGrid<Dataset>>(); }
        else { return std::make_unique<SedovGlass<Dataset>>(glassBlock); }
    }
    if (testCase == "noh")
    {
        if (glassBlock.empty()) { return std::make_unique<NohGrid<Dataset>>(); }
        else { return std::make_unique<NohGlassSphere<Dataset>>(glassBlock); }
    }
    if (testCase == "isobaric-cube")
    {
        if (glassBlock.empty()) { throw std::runtime_error("need a valid glass block for isobaric cube\n"); }
        return std::make_unique<IsobaricCubeGlass<Dataset>>(glassBlock);
    }
    if (testCase == "wind-shock")
    {
        if (glassBlock.empty()) { throw std::runtime_error("need a valid glass block for Wind shock\n"); }
        return std::make_unique<WindShockGlass<Dataset>>(glassBlock);
    }
    if (testCase == "evrard")
    {
        if (glassBlock.empty()) { throw std::runtime_error("need a valid glass block for evrard\n"); }
        return std::make_unique<EvrardGlassSphere<Dataset>>(glassBlock);
    }
    else
    {
        if (std::filesystem::exists(testCase)) { return std::make_unique<FileInit<Dataset>>(testCase); }
        else
        {
            throw std::runtime_error("supplied value of --init " +
                                     (testCase.empty() ? "[empty string]" : "(\"" + testCase + "\")") +
                                     " is not an existing file and does not refer to a supported test case\n");
        }
    }
}

} // namespace sphexa
