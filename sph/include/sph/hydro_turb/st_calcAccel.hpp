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
 * @brief  Implementation of stirring accelerations
 * @author Axel Sanz <axel.sanz@estudiantat.upc.edu>
 */

#pragma once

#include <cmath>

namespace sph
{

/*! @brief Adds the stirring accelerations to the provided particle accelerations
 *
 * @tparam        T                 float or double
 * @param[in]     startIndex        first index of particles
 * @param[in]     endIndex          last index of particles
 * @param[in]     ndim              number of dimensions
 * @param[in]     x                 vector of x components of particle positions
 * @param[in]     y                 vector of y components of particle positions
 * @param[in]     z                 vector of z components of particle positionSs
 * @param[inout]  ax                vector of x component of accelerations
 * @param[inout]  ay                vector of y component of accelerations
 * @param[inout]  az                vector of z component of accelerations
 * @param[in]     st_nmodes         number of modes
 * @param[in]     st_mode           matrix (st_nmodes x dimension) containing modes
 * @param[in]     st_aka            matrix (st_nmodes x dimension) containing real phases
 * @param[in]     st_akb            matrix (st_nmodes x dimension) containing imaginary phases
 * @param[in]     st_ampl           vector of amplitudes of modes
 * @param[in]     st_solweightnorm  normalized solenoidal weight
 *
 */
template<class T>
void st_calcAccel(size_t startIndex, size_t endIndex, size_t ndim, std::vector<T> x, std::vector<T> y, std::vector<T> z,
                  std::vector<T>& ax, std::vector<T>& ay, std::vector<T>& az, size_t st_nmodes, std::vector<T> st_mode,
                  std::vector<T> st_aka, std::vector<T> st_akb, std::vector<T> st_ampl, T st_solweightnorm)
{
#pragma omp parallel for schedule(static)
    for (size_t i = startIndex; i < endIndex; ++i)
    {
        T accturbx = 0.0;
        T accturby = 0.0;
        T accturbz = 0.0;

        for (size_t m = 0; m < st_nmodes; ++m)
        {
            size_t m_ndim = m * ndim;

            T cosxk_im = std::cos(st_mode[m_ndim + 2] * z[i]);
            T sinxk_im = std::sin(st_mode[m_ndim + 2] * z[i]);

            T cosxj_im = std::cos(st_mode[m_ndim + 1] * y[i]);
            T sinxj_im = std::sin(st_mode[m_ndim + 1] * y[i]);

            T cosxi_im = std::cos(st_mode[m_ndim] * x[i]);
            T sinxi_im = std::sin(st_mode[m_ndim] * x[i]);

            //  these are the real and imaginary parts, respectively, of
            //     e^{ i \vec{k} \cdot \vec{x} }
            //          = cos(kx*x + ky*y + kz*z) + i sin(kx*x + ky*y + kz*z)
            T realtrigterms = (cosxi_im * cosxj_im - sinxi_im * sinxj_im) * cosxk_im -
                              (sinxi_im * cosxj_im + cosxi_im * sinxj_im) * sinxk_im;

            T imtrigterms = cosxi_im * (cosxj_im * sinxk_im + sinxj_im * cosxk_im) +
                            sinxi_im * (cosxj_im * cosxk_im - sinxj_im * sinxk_im);

            accturbx += st_ampl[m] * (st_aka[m_ndim] * realtrigterms - st_akb[m_ndim] * imtrigterms);
            accturby += st_ampl[m] * (st_aka[m_ndim + 1] * realtrigterms - st_akb[m_ndim + 1] * imtrigterms);
            accturbz += st_ampl[m] * (st_aka[m_ndim + 2] * realtrigterms - st_akb[m_ndim + 2] * imtrigterms);
        }

        ax[i] += st_solweightnorm * accturbx;
        ay[i] += st_solweightnorm * accturby;
        az[i] += st_solweightnorm * accturbz;
    }
}

} // namespace sph
