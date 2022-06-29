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

#include <cmath>
#include <iostream>

namespace sph{

template<class T>
void st_calcAccel(size_t first, size_t last,size_t ndim, std::vector<T> xCoord,std::vector<T> yCoord,std::vector<T> zCoord,
  std::vector<T>& accx,std::vector<T>& accy,std::vector<T>& accz, size_t st_nmodes,
  std::vector<T> st_mode,std::vector<T> st_aka,std::vector<T> st_akb,std::vector<T> st_ampl,T st_solweightnorm){

  const size_t npart=last-first;
  T cosxi[npart][st_nmodes], sinxi[npart][st_nmodes];
  T cosxj[npart][st_nmodes], sinxj[npart][st_nmodes];
  T cosxk[npart][st_nmodes], sinxk[npart][st_nmodes];
  //T ampl[st_nmodes];
  T accturbx[npart], accturby[npart], accturbz[npart];

  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < npart; ++i){
    size_t i_first = i + first;

    for (size_t m = 0; m < st_nmodes ; ++m){
       size_t m_ndim  = m * ndim;

       cosxk[i][m] = std::cos(st_mode[m_ndim + 2] * zCoord[i_first]);
       sinxk[i][m] = std::sin(st_mode[m_ndim + 2] * zCoord[i_first]);

       cosxj[i][m] = std::cos(st_mode[m_ndim + 1] * yCoord[i_first]);
       sinxj[i][m] = std::sin(st_mode[m_ndim + 1] * yCoord[i_first]);

       cosxi[i][m] = std::cos(st_mode[m_ndim] * xCoord[i_first]);
       sinxi[i][m] = std::sin(st_mode[m_ndim] * xCoord[i_first]);

    }
  }
// Loop over particles

  #pragma omp parallel for schedule(static)
      for (size_t i = 0; i < npart; i++){
           accturbx[i] = 0.0;
           accturby[i] = 0.0;
           accturbz[i] = 0.0;
           size_t i_first = i + first;
           for (size_t m = 0; m < st_nmodes; m++){
                size_t m_ndim = m * ndim;
                //  these are the real and imaginary parts, respectively, of
                //     e^{ i \vec{k} \cdot \vec{x} }
                //          = cos(kx*x + ky*y + kz*z) + i sin(kx*x + ky*y + kz*z)

                T realtrigterms = ( cosxi[i][m] * cosxj[i][m] - sinxi[i][m] * sinxj[i][m] ) * cosxk[i][m]
                                - ( sinxi[i][m] * cosxj[i][m] + cosxi[i][m] * sinxj[i][m] ) * sinxk[i][m];

                T imtrigterms   =  cosxi[i][m] * ( cosxj[i][m] * sinxk[i][m] + sinxj[i][m] * cosxk[i][m] )
                                +  sinxi[i][m] * ( cosxj[i][m] * cosxk[i][m] - sinxj[i][m] * sinxk[i][m] );

                accturbx[i] += st_ampl[m] * (st_aka[m_ndim] * realtrigterms - st_akb[m_ndim] * imtrigterms);
                accturby[i] += st_ampl[m] * (st_aka[m_ndim + 1] * realtrigterms - st_akb[m_ndim + 1] * imtrigterms);
                accturbz[i] += st_ampl[m] * (st_aka[m_ndim + 2] * realtrigterms - st_akb[m_ndim + 2] * imtrigterms);
                //std::cout  << st_aka[m_ndim] << ' ' << st_aka[m_ndim+1] << ' ' << st_aka[m_ndim+2] << std::endl;
           }  // end loop over modes

           accx[i_first] += st_solweightnorm * accturbx[i];
           accy[i_first] += st_solweightnorm * accturby[i];
           accz[i_first] += st_solweightnorm * accturbz[i];
      }  // end loop over particles

  return;

}
} //namespace sphexa
