/*
 * MIT License
 *
 * Copyright (c) 2022 Politechnical University of Catalonia UPC
 *               2022 University of Basel
 *               2022 CSCS, ETH Zurich
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
 * @brief  st_calcAccel: Adds the stirring accelerations to the provided accelerations
 *           Input Arguments:
 *             npart:                  number of particles
 *             xCoord:                 vector of x components of particle positions
 *             yCoord:                 vector of y components of particle positions
 *             zCoord:                 vector of z components of particle positions
 *             st_nmodes:              number of modes
 *             st_mode:                matrix (st_nmodes x dimension) containing modes
 *             st_aka:                 matrix (st_nmodes x dimension) containing real phases
 *             st_akb:                 matrix (st_nmodes x dimension) containing imaginary phases
 *             st_ampl:                vector of amplitudes of modes
 *             st_solweightnorm:       normalized solenoidal weight
 *           Input/Output Arguments:
 *             accx:                   vector of x component of accelerations
 *             accy:                   vector of y component of accelerations
 *             accz:                   vector of z component of accelerations
 * @author Axel Sanz <axel.sanz@estudiantat.upc.edu>
 */

#pragma once
#include <cmath>

namespace sph
{

/*! @brief  uniform random number generator in range (0,1)
 *           Input/Output Arguments:
 *             idum:                   seed of the generator
 *            Returns:
 *              rand:                   random number
 */
template<class T>
T ran1s(long int& idum)
{

    int       k, iy;
    const int IA = 16807, IM = 2147483647, IQ = 127773, IR = 2836;
    const T   AM = 1. / IM, EPS = 1.2e-7, RNMX = 1. - EPS;
    T         rand;

    if (idum <= 0) { idum = std::max(-idum, long(1)); }
    k    = idum / IQ;
    idum = IA * (idum - k * IQ) - IR * k;
    if (idum < 0) { idum = idum + IM; }
    iy   = idum;
    rand = std::min(AM * iy, RNMX);
    return rand;
}

/*! @brief  ran1s: guassian random number generator  with unit variance and mean 0
 *           Input/Output Arguments:
 *             idum:                   seed of the generator
 *           Returns:
 *             rand:                   random number
 */
template<class T>
T st_grn(long int& seed)
{
    T       r1, r2, g1;
    const T twopi = 2.0 * M_PI;

    r1 = ran1s<T>(seed);
    r2 = ran1s<T>(seed);
    g1 = std::sqrt(-2.0 * std::log(r1)) * std::cos(twopi * r2);

    return g1;
}

/*! @brief Generates an Ornstein-Uhlenbeck sequence.
 *
 *   @param[inout] phases        the Ornstein-Uhlenbeck phases to be updated
 *   @param[in] variance      variance of the distribution
 *   @param[in] dt            timestep
 *   @param[in] ts            autocorrelation time
 *
 * The sequence x_n is a Markov process that takes the previous value,
 *   weights by an exponential damping factor with a given correlation
 *   time "ts", and drives by adding a Gaussian random variable with
 *   variance "variance", weighted by a second damping factor, also
 *   with correlation time "ts". For a timestep of dt, this sequence
 *   can be written as :
 *
 *     x_n+1 = f x_n + sigma * sqrt (1 - f**2) z_n
 *
 * where f = exp (-dt / ts), z_n is a Gaussian random variable drawn
 * from a Gaussian distribution with unit variance, and sigma is the
 * desired variance of the OU sequence. (See Bartosch, 2001).
 *
 * The resulting sequence should satisfy the properties of zero mean,
 *   and stationary (independent of portion of sequence) RMS equal to
 *   "variance". Its power spectrum in the time domain can vary from
 *   white noise to "brown" noise (P (f) = const. to 1 / f^2).
 *
 * References :
 *    Bartosch, 2001
 * http://octopus.th.physik.uni-frankfurt.de/~bartosch/publications/IntJMP01.pdf
 *   Finch, 2004
 * http://pauillac.inria.fr/algo/csolve/ou.pdf
 *         Uhlenbeck & Ornstein, 1930
 * http://prola.aps.org/abstract/PR/v36/i5/p823_1
 *
 * Eswaran & Pope 1988
 *
 */
template<class T>
void st_ounoiseupdate(std::vector<T>& phases, T variance, T dt, T ts, long int& seed)
{
    T damping_factor;
    damping_factor = std::exp(-dt / ts);
    for (size_t i = 0; i < phases.size(); i++)
    {
        phases[i] =
            phases[i] * damping_factor + variance * sqrt(1.0 - damping_factor * damping_factor) * st_grn<T>(seed);
    }
}

//! @brief fills a vector with random gaussian values with mean 0 and variance @p variance
template<class T>
void fillRandomGaussian(std::vector<T>& phases, T variance, long int& seed)
{
    for (size_t i = 0; i < phases.size(); i++)
    {
        phases[i] = st_grn<T>(seed) * variance;
    }
}

} // namespace sph
