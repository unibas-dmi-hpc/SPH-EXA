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
  size_t i_first,m_ndim;
  T cosxi[st_nmodes][npart], sinxi[st_nmodes][npart];
  T cosxj[st_nmodes][npart], sinxj[st_nmodes][npart];
  T cosxk[st_nmodes][npart], sinxk[st_nmodes][npart];
  T realtrigterms, imtrigterms;
  size_t ib, ie;
  size_t i, j, k, m, iii;
  T ampl[st_nmodes];
  T accturbx[npart], accturby[npart], accturbz[npart];

  //$omp parallel private (m)
  //$omp do schedule(static) */
  for(m = 0; m<st_nmodes;m++){
    for(i = 0; i<npart; i++){
       i_first=i+first;
       m_ndim=m*ndim;
       cosxk[m][i] = std::cos(st_mode[m_ndim+2]*zCoord[i_first]);
       sinxk[m][i] = std::sin(st_mode[m_ndim+2]*zCoord[i_first]);

       cosxj[m][i] = std::cos(st_mode[m_ndim+1]*yCoord[i_first]);
       sinxj[m][i] = std::sin(st_mode[m_ndim+1]*yCoord[i_first]);

       cosxi[m][i] = std::cos(st_mode[m_ndim]*xCoord[i_first]);
       sinxi[m][i] = std::sin(st_mode[m_ndim]*xCoord[i_first]);

    }
    //// save some cpu time by precomputing one more multiplication
    ampl[m] = 2.0*st_ampl[m];
  }

  //$omp end do
  //$omp end parallel

//   ib = blkLimitsGC(LOW,IAXIS)
//   ie = blkLimitsGC(HIGH,IAXIS)
//
//   call Grid_getBlkPtr(blockID,solnData)
//
//   //// loop over all cells
//   do k = blkLimitsGC (LOW, KAXIS), blkLimitsGC(HIGH,KAXIS)
//      do j = blkLimitsGC (LOW, JAXIS), blkLimitsGC(HIGH,JAXIS)
//
// #ifdef ACCX_VAR
//         solnData(ACCX_VAR,ib:ie,j,k) = 0.0
// #else
//         accx(ib:ie,j,k) = 0.0
// #endif
// #ifdef ACCY_VAR
//         solnData(ACCY_VAR,ib:ie,j,k) = 0.0
// #else
//         accy(ib:ie,j,k) = 0.0
// #endif
// #ifdef ACCZ_VAR
//         solnData(ACCZ_VAR,ib:ie,j,k) = 0.0
// #else
//         accz(ib:ie,j,k) = 0.0
// #endif
//
//         do i = blkLimitsGC (LOW, IAXIS), blkLimitsGC(HIGH,IAXIS)

// Loop over particles

      //$omp parallel private(i,m,realtrigterms,imtrigterms)
      //$omp do schedule(static)
      for(i=0;i<npart;i++){
        accturbx[i]=0.0;
        accturby[i]=0.0;
        accturbz[i]=0.0;
        i_first=i+first;
           for(m = 0; m<st_nmodes; m++){
                m_ndim=m*ndim;
                //  these are the real and imaginary parts, respectively, of
                //     e^{ i \vec{k} \cdot \vec{x} }
                //          = cos(kx*x + ky*y + kz*z) + i sin(kx*x + ky*y + kz*z)

                realtrigterms =    ( cosxi[m][i]*cosxj[m][i] - sinxi[m][i]*sinxj[m][i] ) * cosxk[m][i]
                                 - ( sinxi[m][i]*cosxj[m][i] + cosxi[m][i]*sinxj[m][i] ) * sinxk[m][i];

                imtrigterms   =    cosxi[m][i] * ( cosxj[m][i]*sinxk[m][i] + sinxj[m][i]*cosxk[m][i] )
                                 + sinxi[m][i] * ( cosxj[m][i]*cosxk[m][i] - sinxj[m][i]*sinxk[m][i] );

                accturbx[i]   = accturbx[i] + ampl[m]*(st_aka[m_ndim]*realtrigterms-st_akb[m_ndim]*imtrigterms);
                accturby[i]   = accturby[i] + ampl[m]*(st_aka[m_ndim+1]*realtrigterms-st_akb[m_ndim+1]*imtrigterms);
                accturbz[i]   = accturbz[i] + ampl[m]*(st_aka[m_ndim+2]*realtrigterms-st_akb[m_ndim+2]*imtrigterms);
                //std::cout  << st_aka[m_ndim] << ' ' << st_aka[m_ndim+1] << ' ' << st_aka[m_ndim+2] << std::endl;
           }  // end loop over modes

           accx[i_first] += st_solweightnorm*accturbx[i];
           accy[i_first] += st_solweightnorm*accturby[i];
           accz[i_first] += st_solweightnorm*accturbz[i];
      }  // end loop over particles
      //$omp end do
      //$omp end parallel

     // accturb(:)=0.d0;
      //for( i=0; i<npart; i++){
       // iii=i+npini;
        //accturb[iii]=accx[i];
        //accturb[iii+n]=accy[i];
        //accturb[iii+n2]=accz[i];
      //}

      //for (i=npini;i<=npend;i++){
        //accturb[npini:npend]=accturb[npini:npend]*st_solweightnorm;
      //}
  return;

}
} //namespace sphexa
