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
void st_calcAccel(const int npart,double xCoord[],double yCoord[],double zCoord[],double accx[],double accy[],double accz[],
                  int st_nmodes,double st_mode[][3],double st_aka[][3],double st_akb[][3],double st_ampl[],double st_solweightnorm){

  double cosxi[st_nmodes][npart], sinxi[st_nmodes][npart];
  double cosxj[st_nmodes][npart], sinxj[st_nmodes][npart];
  double cosxk[st_nmodes][npart], sinxk[st_nmodes][npart];
  double realtrigterms, imtrigterms;
  int ib, ie;
  int i, j, k, m, iii;
  double ampl[st_nmodes];

  //$omp parallel private (m)
  //$omp do schedule(static) */
  for( m = 0; m<st_nmodes;m++){
    for( i = 0; i<npart; i++){
  
       cosxk[m][i] = std::cos(st_mode[m][2]*zCoord[i]);
       sinxk[m][i] = std::sin(st_mode[m][2]*zCoord[i]);
  
       cosxj[m][i] = std::cos(st_mode[m][1]*yCoord[i]);            //HELP (array operations?) HAcer loop
       sinxj[m][i] = std::sin(st_mode[m][1]*yCoord[i]);
  
       cosxi[m][i] = std::cos(st_mode[m][0]*xCoord[i]);
       sinxi[m][i] = std::sin(st_mode[m][0]*xCoord[i]);
  
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
        //accx[i]=0.0;
        //accy[i]=0.0;
        //accz[i]=0.0;
        
           for(m = 0; m<st_nmodes; m++){

                //  these are the real and imaginary parts, respectively, of
                //     e^{ i \vec{k} \cdot \vec{x} }
                //          = cos(kx*x + ky*y + kz*z) + i sin(kx*x + ky*y + kz*z)

                realtrigterms =    ( cosxi[m][i]*cosxj[m][i] - sinxi[m][i]*sinxj[m][i] ) * cosxk[m][i]  
                                 - ( sinxi[m][i]*cosxj[m][i] + cosxi[m][i]*sinxj[m][i] ) * sinxk[m][i];
                                 
                imtrigterms   =    cosxi[m][i] * ( cosxj[m][i]*sinxk[m][i] + sinxj[m][i]*cosxk[m][i] ) 
                                 + sinxi[m][i] * ( cosxj[m][i]*cosxk[m][i] - sinxj[m][i]*sinxk[m][i] );
                                 
                accx[i]   = accx[i] + ampl[m]*(st_aka[m][0]*realtrigterms-st_akb[m][0]*imtrigterms);
                accy[i]   = accy[i] + ampl[m]*(st_aka[m][1]*realtrigterms-st_akb[m][1]*imtrigterms);
                accz[i]   = accz[i] + ampl[m]*(st_aka[m][2]*realtrigterms-st_akb[m][2]*imtrigterms);

           }  // end loop over modes

           accx[i] = st_solweightnorm*accx[i];
           accy[i] = st_solweightnorm*accy[i];
           accz[i] = st_solweightnorm*accz[i];
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
