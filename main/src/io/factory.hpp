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

#include <fstream>
#include <map>
#include <memory>

#include "cstone/sfc/box.hpp"

namespace sphexa
{

template<class Dataset>
std::unique_ptr<IObservables<Dataset>> observablesFactory(std::string testCase, std::ofstream constantsFile)
{
    if (testCase == "IC_kelvin-helmholtz.h5")
    {
        return std::make_unique<TimeEnergyGrowth<Dataset>>(constantsFile);
    }
    else
    {
       //return std::make_unique<TimeAndEnergy<Dataset>>(constantsFile);
    }
}

} // namespace sphexa
