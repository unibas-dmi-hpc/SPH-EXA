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
 * @brief Contains the object holding all stirring/turbulence related data
 *
 * @author Axel Sanz <axelsanzlechuga@gmail.com>*
 */

#pragma once

#include <cstdint>
#include <vector>

namespace sph
{

template<class T>
class TurbulenceData
{
public:
    long int stSeed;          // seed for random number generator
    size_t   stNModes;        // Number of computed nodes
    size_t   ndim;            // Number of dimensions
    T        stSolWeight;     // Solenoidal Weight
    T        stOUvar;         // Variance of Ornstein-Uhlenbeck process
    T        stDecay;         // Decay Time
    T        stSolWeightNorm; // Normalized Solenoidal weight

    std::vector<T> stOUPhases; // O-U Phases
    std::vector<T> stMode;     // Stirring Modes
    std::vector<T> stAmpl;     // Amplitude of the modes
};

} // namespace sph
