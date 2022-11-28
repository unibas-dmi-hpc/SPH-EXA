/*
 * MIT License
 *
 * Copyright (c) 2022 CSCS, ETH Zurich, University of Zurich, University of Basel
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
 * @brief Unit tests for ParticlesData
 *
 * @author Noah Kubli <noah.kubli@uzh.ch>
 */

#pragma once

#include "evrard_init.hpp"
#include "cooling/cooling.hpp"
#include "cooling/cooler.h"

template<class Dataset>
class EvrardGlassSphereCooling : public sphexa::EvrardGlassSphere<Dataset>
{
    const float ms_sim = 1e16;
    const float kp_sim = 46400.;

public:
    EvrardGlassSphereCooling(std::string initBlock)
        : sphexa::EvrardGlassSphere<Dataset>(initBlock)
    {
    }
    cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, size_t cbrtNumPart,
                                                 Dataset& simData) const override
    {
        cooling::chemistry_data_ grackleData = simData.chem.cooling_data.getDefaultChemistryData();

        grackleData.content.use_grackle            = 1;
        grackleData.content.with_radiative_cooling = 1;
        grackleData.content.primordial_chemistry   = 0;
        grackleData.content.dust_chemistry         = 0;
        grackleData.content.metal_cooling          = 0;
        grackleData.content.UVbackground           = 0;

        simData.chem.cooling_data.init(ms_sim, kp_sim, 0, grackleData, std::nullopt, std::nullopt);

        auto box = sphexa::EvrardGlassSphere<Dataset>::init(rank, numRanks, cbrtNumPart, simData);
        cooling::initGrackleData(simData.chem, simData.hydro.x.size());
        return box;
    }
};
