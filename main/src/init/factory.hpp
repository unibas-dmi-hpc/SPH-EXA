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

#include "io/arg_parser.hpp"
#include "isim_init.hpp"
#include "sedov_init.hpp"
#include "evrard_init.hpp"
#include "file_init.hpp"
#include "isobaric_cube_init.hpp"
#include "kelvin_helmholtz_init.hpp"
#include "noh_init.hpp"
#include "turbulence_init.hpp"
#include "wind_shock_init.hpp"
#ifdef SPH_EXA_HAVE_GRACKLE
#include "evrard_cooling_init.hpp"
#endif

namespace sphexa
{

template<class Dataset>
std::unique_ptr<ISimInitializer<Dataset>> initializerFactory(std::string testCase, std::string glassBlock,
                                                             IFileReader* reader)
{
    std::string testNamedBase = strBeforeSign(testCase, ":");
    std::string settingsFile  = strAfterSign(testCase, ":");

    if (testNamedBase == "sedov")
    {
        if (glassBlock.empty()) { return std::make_unique<SedovGrid<Dataset>>(); }
        else { return std::make_unique<SedovGlass<Dataset>>(glassBlock, settingsFile, reader); }
    }
    if (testNamedBase == "noh")
    {
        if (glassBlock.empty()) { throw std::runtime_error("need a valid glass block for Noh implosion\n"); }
        return std::make_unique<NohGlassSphere<Dataset>>(glassBlock, settingsFile, reader);
    }
    if (testNamedBase == "isobaric-cube")
    {
        if (glassBlock.empty()) { throw std::runtime_error("need a valid glass block for isobaric cube\n"); }
        return std::make_unique<IsobaricCubeGlass<Dataset>>(glassBlock, settingsFile, reader);
    }
    if (testNamedBase == "wind-shock")
    {
        if (glassBlock.empty()) { throw std::runtime_error("need a valid glass block for Wind shock\n"); }
        return std::make_unique<WindShockGlass<Dataset>>(glassBlock, settingsFile, reader);
    }
    if (testNamedBase == "evrard")
    {
        if (glassBlock.empty()) { throw std::runtime_error("need a valid glass block for evrard\n"); }
        return std::make_unique<EvrardGlassSphere<Dataset>>(glassBlock, settingsFile, reader);
    }
    if (testNamedBase == "turbulence")
    {
        if (glassBlock.empty()) { throw std::runtime_error("need a valid glass block for turbulence test\n"); }
        else { return std::make_unique<TurbulenceGlass<Dataset>>(glassBlock, settingsFile, reader); }
    }
    if (testNamedBase == "kelvin-helmholtz")
    {
        if (glassBlock.empty()) { throw std::runtime_error("need a valid glass block for Kelvin-Helmholtz test\n"); }
        else { return std::make_unique<KelvinHelmholtzGlass<Dataset>>(glassBlock, settingsFile, reader); }
    }
#ifdef SPH_EXA_HAVE_GRACKLE
    if (testCase == "evrard-cooling")
    {
        if (glassBlock.empty()) { throw std::runtime_error("need a valid glass block for evrard-cooling\n"); }
        return std::make_unique<EvrardGlassSphereCooling<Dataset>>(glassBlock, settingsFile, reader);
    }
#endif
    if (std::filesystem::exists(strBeforeSign(testCase, ":")))
    {
        return std::make_unique<FileInit<Dataset>>(testCase, reader);
    }
    if (std::filesystem::exists(strBeforeSign(testCase, ",")))
    {
        return std::make_unique<FileSplitInit<Dataset>>(testCase, reader);
    }

    auto msg = "supplied value of --init " + (testCase.empty() ? "[empty string]" : "(\"" + testCase + "\")") +
               " is not a usable file or supported test case\n";
    if (reader->rank() == 0) { std::cout << msg; }
    MPI_Abort(MPI_COMM_WORLD, 1);

    return std::unique_ptr<ISimInitializer<Dataset>>{};
}

} // namespace sphexa
