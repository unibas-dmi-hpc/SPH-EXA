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
////                                   It copies them over and applies the projection operator.
 *           Input Arguments:
 *             st_nmodes:              computed number of modes
 *             dim:                    number of dimensions
 *             st_OUphases:            Ornstein-Uhlenbeck phases
 *             st_solweight:           solenoidal weight
 *             st_mode:                matrix (st_nmodes x dimension) containing modes        
 *           Output Arguments:
 *             st_aka:                 real part of phases
 *             st_akb:                 imaginary part of phases
 * @author Axel Sanz <axel.sanz@estudiantat.upc.edu>
 */
#include <cmath>
#include <iostream>
void st_calcPhases(int st_nmodes,int dim,double st_OUphases[],double st_solweight,double st_mode[][3],double st_aka[][3], double st_akb[][3]){

  double ka, kb, kk, diva, divb, curla, curlb;
  int i,j;
  const bool Debug = false;

  for(i = 0; i< st_nmodes;i++){
     ka = 0.0;
     kb = 0.0;
     kk = 0.0;
     for(j = 0; j< dim;j++){
        kk = kk + st_mode[i][j]*st_mode[i][j];
        ka = ka + st_mode[i][j]*st_OUphases[6*(i)+2*(j)+1];
        kb = kb + st_mode[i][j]*st_OUphases[6*(i)+2*(j)+0];
     }
     for(j = 0; j< dim;j++){

         diva  = st_mode[i][j]*ka/kk;
         divb  = st_mode[i][j]*kb/kk;
         curla = (st_OUphases[6*(i)+2*(j) + 0 ] - divb);
         curlb = (st_OUphases[6*(i)+2*(j) + 1 ] - diva);

         st_aka[i][j] = st_solweight*curla+(1.0-st_solweight)*divb;
         st_akb[i][j] = st_solweight*curlb+(1.0-st_solweight)*diva;
         
      }

// purely compressive
//         st_aka[i][j] = st_mode[i][j]*kb/kk
//         st_akb[i][j] = st_mode[i][j]*ka/kk

// purely solenoidal
//         st_aka[i][j] = bjiR - st_mode[i][j]*kb/kk
//         st_akb[i][j] = bjiI - st_mode[i][j]*ka/kk


     }
  
  return;

}
