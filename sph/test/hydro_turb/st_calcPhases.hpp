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
#pragma once
#include <cmath>
#include <iostream>
#include <vector>

template <class T>
void st_calcPhases(size_t st_nmodes, size_t dim, std::vector<T> st_OUphases, T st_solweight,
  std::vector<T> st_mode, std::vector<T>& st_aka, std::vector<T>& st_akb){

  T ka, kb, kk, diva, divb, curla, curlb;
  size_t i,j;
  const bool Debug = false;

  for(i = 0; i< st_nmodes;i++){
     ka = 0.0;
     kb = 0.0;
     kk = 0.0;
     for(j = 0; j< dim;j++){
        kk = kk + st_mode[3 * i + j] * st_mode[3 * i + j];
        ka = ka + st_mode[3 * i + j] * st_OUphases[6 * i + 2 * j + 1];
        kb = kb + st_mode[3 * i + j] * st_OUphases[6 * i + 2 * j];
     }
     for(j = 0; j< dim;j++){

         diva  = st_mode[3 * i + j] * ka / kk;
         divb  = st_mode[3 * i + j] * kb / kk;
         curla = st_OUphases[6 * i + 2 * j] - divb;
         curlb = st_OUphases[6 * i + 2 * j + 1] - diva;

         st_aka[3 * i + j] = st_solweight * curla + ( 1.0 - st_solweight ) * divb;
         st_akb[3 * i + j] = st_solweight * curlb + ( 1.0 - st_solweight ) * diva;

      }

// purely compressive
//         st_aka[3*i+j] = st_mode[3*i+j]*kb/kk
//         st_akb[3*i+j] = st_mode[3*i+j]*ka/kk

// purely solenoidal
//         st_aka[3*i+j] = bjiR - st_mode[3*i+j]*kb/kk
//         st_akb[3*i+j] = bjiI - st_mode[3*i+j]*ka/kk

     }
  return;

}
