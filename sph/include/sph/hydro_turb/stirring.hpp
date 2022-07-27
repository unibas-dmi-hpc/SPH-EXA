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

template<class Tc, class T>
CUDA_DEVICE_HOST_FUN auto stirParticle(size_t ndim, Tc xi, Tc yi, Tc zi, size_t numModes, const T* modes,
                                       const T* st_aka, const T* st_akb, const T* st_ampl)

{
    T turbAx = 0.0;
    T turbAy = 0.0;
    T turbAz = 0.0;

    for (size_t m = 0; m < numModes; ++m)
    {
        size_t m_ndim = m * ndim;

        T cosxk_im = std::cos(modes[m_ndim + 2] * zi);
        T sinxk_im = std::sin(modes[m_ndim + 2] * zi);

        T cosxj_im = std::cos(modes[m_ndim + 1] * yi);
        T sinxj_im = std::sin(modes[m_ndim + 1] * yi);

        T cosxi_im = std::cos(modes[m_ndim] * xi);
        T sinxi_im = std::sin(modes[m_ndim] * xi);

        //  these are the real and imaginary parts, respectively, of
        //     e^{ i \vec{k} \cdot \vec{x} }
        //          = cos(kx*x + ky*y + kz*z) + i sin(kx*x + ky*y + kz*z)
        T realtrigterms = (cosxi_im * cosxj_im - sinxi_im * sinxj_im) * cosxk_im -
                          (sinxi_im * cosxj_im + cosxi_im * sinxj_im) * sinxk_im;

        T imtrigterms = cosxi_im * (cosxj_im * sinxk_im + sinxj_im * cosxk_im) +
                        sinxi_im * (cosxj_im * cosxk_im - sinxj_im * sinxk_im);

        turbAx += st_ampl[m] * (st_aka[m_ndim] * realtrigterms - st_akb[m_ndim] * imtrigterms);
        turbAy += st_ampl[m] * (st_aka[m_ndim + 1] * realtrigterms - st_akb[m_ndim + 1] * imtrigterms);
        turbAz += st_ampl[m] * (st_aka[m_ndim + 2] * realtrigterms - st_akb[m_ndim + 2] * imtrigterms);
    }

    return util::tuple<T, T, T>(turbAx, turbAy, turbAz);
}

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
 * @param[in]     numModes          number of modes
 * @param[in]     modes           matrix (st_nmodes x dimension) containing modes
 * @param[in]     st_aka            matrix (st_nmodes x dimension) containing real phases
 * @param[in]     st_akb            matrix (st_nmodes x dimension) containing imaginary phases
 * @param[in]     st_ampl           vector of amplitudes of modes
 * @param[in]     st_solweightnorm  normalized solenoidal weight
 *
 */
template<class T>
void computeStirring(size_t startIndex, size_t endIndex, size_t ndim, const std::vector<T>& x, const std::vector<T>& y,
                     const std::vector<T>& z, std::vector<T>& ax, std::vector<T>& ay, std::vector<T>& az,
                     size_t numModes, const std::vector<T>& modes, const std::vector<T>& st_aka,
                     const std::vector<T>& st_akb, const std::vector<T>& st_ampl, T st_solweightnorm)
{
#pragma omp parallel for schedule(static)
    for (size_t i = startIndex; i < endIndex; ++i)
    {
        auto [turbAx, turbAy, turbAz] =
            stirParticle(ndim, x[i], y[i], z[i], numModes, modes.data(), st_aka.data(), st_akb.data(), st_ampl.data());

        ax[i] += st_solweightnorm * turbAx;
        ay[i] += st_solweightnorm * turbAy;
        az[i] += st_solweightnorm * turbAz;
    }
}

} // namespace sph
