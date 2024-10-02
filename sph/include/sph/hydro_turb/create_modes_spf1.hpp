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
 * @brief Turbulence simulation data initialization
 *
 * @author Axel Sanz <axelsanzlechuga@gmail.com>
 */

#pragma once

#include <iostream>

#include "sph/hydro_turb/turbulence_data.hpp"

namespace sph
{

/*! @brief TODO: Short description of this function
 *
 * @tparam       T                 float or double
 * @param[inout] d                 the turbulence dataset
 * @param[in]    Lx
 * @param[in]    Ly
 * @param[in]    Lz
 * @param[in]    st_maxmodes
 * @param[in]    stirMax
 * @param[in]    stirMin
 * @param[in]    ndim
 * @param[in]    spectForm
 * @param[in]    powerLawExp
 * @param[in]    anglesExp
 *
 * Function description missing
 */

template<class T>
void createStirringModesSpf1(int &st_tot_nmodes, T modes[], T amplitudes[], T Lx, T Ly, T Lz, size_t st_maxmodes, T stirMax, T stirMin, size_t ndim,
                          bool verbose)
{
    // TODO: this should not take the whole Dataset, only modes and amplitudes are needed
    // TODO: st_maxmodes should not be an input parameter. the caller will have determineNumModes
    // TODO: and it's up to the call-site to decide if they want to create as many as it takes (or skip the check)

    const T twopi = 2.0 * M_PI;

    // characteristic k for scaling the amplitude below
    
    T kc = 0.5 * (stirMin + stirMax);

    size_t ikxmin = 0;
    size_t ikymin = 0;
    size_t ikzmin = 0;

    size_t ikxmax = 256;
    size_t ikymax = (ndim > 1) ? 256 : 0;
    size_t ikzmax = (ndim > 2) ? 256 : 0;

    // TODO: this should be a separate function "determineNumModes".
    // determine the number of required modes (in case of full sampling)
    int numModes = 0;
    for (size_t ikx = ikxmin; ikx <= ikxmax; ikx++)
    {
        T kx = twopi * ikx / Lx;
        for (size_t iky = ikymin; iky <= ikymax; iky++)
        {
            T ky = twopi * iky / Ly;
            for (size_t ikz = ikzmin; ikz <= ikzmax; ikz++)
            {
                T kz = twopi * ikz / Lz;
                T k  = std::sqrt(kx * kx + ky * ky + kz * kz);
                if (k >= stirMin && k <= stirMax)
                {
                    numModes += 1;
                    if (ndim > 1) { numModes += 1; }
                    if (ndim > 2) { numModes += 2; }
                }
            }
        }
    }
    st_tot_nmodes = numModes;

    numModes = -1;

           if (verbose) std::cout << "Generating " << st_tot_nmodes << " driving modes..." << std::endl;

        // prefactor for amplitude normalistion to 1 at kc = 0.5*(st_stirmin+st_stirmax)
        T parab_prefact = -4.0 / ((stirMax - stirMin) * (stirMax - stirMin));

        // for band and parabolic spectrum, use the standard full sampling
        // loop over all kx, ky, kz to generate driving modes
        for (size_t ikx = ikxmin; ikx <= ikxmax; ikx++)
        {
            T kx = twopi * ikx / Lx;
            for (size_t iky = ikymin; iky <= ikymax; iky++)
            {
                T ky = twopi * iky / Ly;
                for (size_t ikz = ikzmin; ikz <= ikzmax; ikz++)
                {
                    T kz = twopi * ikz / Lz;
                    T k  = std::sqrt(kx * kx + ky * ky + kz * kz);

                    if ((k >= stirMin) && (k <= stirMax))
                    {
                        if ((numModes + 1 + std::pow(2, ndim - 1)) > st_maxmodes)
                        {
                            std::cout << "init_stir:  number of modes: = " << numModes + 1
                                      << " maxstirmodes = " << st_maxmodes << std::endl;
                            std::cout << "Too many stirring modes" << std::endl;
                            break;
                        }

                        T amplitude = 1.0; // Band
                        
                            amplitude = std::abs(parab_prefact * (k - kc) * (k - kc) + 1.0); // Parabola

                        // note: power spectrum ~ amplitude^2 (1D), amplitude^2 * 2pi k (2D), amplitude^2 * 4pi k^2 (3D)
                        amplitude = 2.0 * std::sqrt(amplitude) * std::pow((kc / k), 0.5 * (ndim - 1));

                        numModes += 1;
                        amplitudes[numModes] = amplitude;

                        modes[ndim * numModes]     = kx;
                        modes[ndim * numModes + 1] = ky;
                        modes[ndim * numModes + 2] = kz;

                        if (ndim > 1)
                        {
                            numModes += 1;
                            amplitudes[numModes] = amplitude;

                            modes[ndim * numModes]     = kx;
                            modes[ndim * numModes + 1] = -ky;
                            modes[ndim * numModes + 2] = kz;
                        }

                        if (ndim > 2)
                        {
                            numModes += 1;
                            amplitudes[numModes] = amplitude;

                            modes[ndim * numModes]     = kx;
                            modes[ndim * numModes + 1] = ky;
                            modes[ndim * numModes + 2] = -kz;

                            numModes += 1;
                            amplitudes[numModes] = amplitude;

                            modes[ndim * numModes]     = kx;
                            modes[ndim * numModes + 1] = -ky;
                            modes[ndim * numModes + 2] = -kz;
                        }
                    } // in k range
                }     // ikz
            }         // iky
        }             // ikx
    }

} // namespace sph
