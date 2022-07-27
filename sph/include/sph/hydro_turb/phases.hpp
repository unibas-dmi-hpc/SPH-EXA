/*
 * MIT License
 *
 * Copyright (c) 2022 Politechnical University of Catalonia UPC
 *               2022 University of Basel
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
 * @brief  st_calcPhases:              This routine updates the stirring phases from the OU phases.
 *                                     It copies them over and applies the projection operator.
 *           Input Arguments:
 *             st_nmodes:              computed number of modes
 *             ndim:                   number of dimensions
 *             st_OUphases:            Ornstein-Uhlenbeck phases
 *             st_solweight:           solenoidal weight
 *             st_mode:                vector containing modes
 *           Output Arguments:
 *             st_aka:                 real part of phases
 *             st_akb:                 imaginary part of phases
 * @author Axel Sanz <axel.sanz@estudiantat.upc.edu>
 */

#pragma once

namespace sph
{

template<class T>
void st_calcPhases(size_t st_nmodes, size_t ndim, const std::vector<T>& st_OUphases, T st_solweight,
                   const std::vector<T>& modes, std::vector<T>& st_aka, std::vector<T>& st_akb)
{
    for (size_t i = 0; i < st_nmodes; i++)
    {
        T ka = 0.0;
        T kb = 0.0;
        T kk = 0.0;
        for (size_t j = 0; j < ndim; j++)
        {
            kk = kk + modes[3 * i + j] * modes[3 * i + j];
            ka = ka + modes[3 * i + j] * st_OUphases[6 * i + 2 * j + 1];
            kb = kb + modes[3 * i + j] * st_OUphases[6 * i + 2 * j];
        }
        for (size_t j = 0; j < ndim; j++)
        {
            T diva  = modes[3 * i + j] * ka / kk;
            T divb  = modes[3 * i + j] * kb / kk;
            T curla = st_OUphases[6 * i + 2 * j] - divb;
            T curlb = st_OUphases[6 * i + 2 * j + 1] - diva;

            st_aka[3 * i + j] = st_solweight * curla + (1.0 - st_solweight) * divb;
            st_akb[3 * i + j] = st_solweight * curlb + (1.0 - st_solweight) * diva;
        }
    }
}

} // namespace sph
