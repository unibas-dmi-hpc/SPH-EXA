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

#include "iobservables.hpp"
#include "time_energy_growth.hpp"
#include "time_energies.hpp"
#include "gravitational_waves.hpp"
#include "wind_bubble_fraction.hpp"
#include "turbulence_mach_rms.hpp"

namespace sphexa
{

template<class Dataset>
std::unique_ptr<IObservables<Dataset>> observablesFactory(const InitSettings& settings, std::ofstream& constantsFile)
{
    if (settings.count("observeGravWaves"))
    {
        if (not settings.count("gravWaveTheta") || not settings.count("graveWavePhi"))
        {
            throw std::runtime_error("need gravWaveTheta ant gravWavePhi input attributes for grav waves observable\n");
        }
        return std::make_unique<GravWaves<Dataset>>(constantsFile, settings.at("gravWaveTheta"),
                                                    settings.at("gravWavePhi"));
    }
    if (settings.count("wind-shock"))
    {
        double rhoInt       = settings.at("rhoInt");
        double uExt         = settings.at("uExt");
        double bubbleVolume = std::pow(settings.at("rSphere"), 3) * 4.0 / 3.0 * M_PI;
        double bubbleMass   = bubbleVolume * rhoInt;
        return std::make_unique<WindBubble<Dataset>>(constantsFile, rhoInt, uExt, bubbleMass);
    }
    if (settings.count("turbulence")) { return std::make_unique<TurbulenceMachRMS<Dataset>>(constantsFile); }
    if (settings.count("kelvin-helmholtz")) { return std::make_unique<TimeEnergyGrowth<Dataset>>(constantsFile); }

    return std::make_unique<TimeAndEnergy<Dataset>>(constantsFile);
}

} // namespace sphexa
