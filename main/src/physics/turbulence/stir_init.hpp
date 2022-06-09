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
 * @brief  stir_init:                  Computes constant parameters, modes and amplitudes of the modes
 *           Input Arguments:
 *             Lx:                     length of box in x direction
 *             Ly:                     length of box in y direction
 *             Lz:                     length of box in z direction
 *             st_maxmodes:            maximum number of modes
 *             st_energy:              characteristic energy
 *             st_decay:               characteristic decay time
 *             st_stirmax:             maximum module of modes wavevector 
 *             st_stirmin:             minimum module of modes wavevector
 *             ndim:                   number of dimensions
 *             st_solweight:           solenoidal weight (typically 0.5)
 *             st_spectform:           
 *           Output Arguments:
 *             st_OUvar:               variance of noise in stirring
 *             st_solweightnorm:       normalized solenoidal weight
 *             st_nmodes:              computed number of modes
               st_ampl:                amplitude of the modes
 *             st_mode:                matrix (st_nmodes x dimension) containing modes
 * @author Axel Sanz <axel.sanz@estudiantat.upc.edu>
 */

#include <cmath>
#include <iostream>
void stir_init(double Lx,double Ly,double Lz,int st_maxmodes,double &st_OUvar,double st_energy,double st_decay,double st_stirmax,double st_stirmin,
              int ndim,double &st_solweightnorm,double st_solweight,int &st_nmodes,double st_ampl[],double st_mode[][3],int st_spectform){
////******************************************************

  //use turbulence_module //uses st_OUvar,st_energy,st_decay,st_stirmax,st_stirmin,ndim,st_solweightnorm,st_solweight,st_nmodes,st_tot_nmodes,st_ampl,st_modes
  //use parameters, only: xbox,ybox,zbox   //HELP (inputs)


  int ikxmin, ikxmax, ikymin, ikymax, ikzmin, ikzmax;
  int ikx, iky, ikz, st_tot_nmodes;
  double kx, ky, kz, k, kc, amplitude, parab_prefact;

  // for uniform random numbers in case of power law (st_spectform .eq. 2)
  int iang, nang, ik, ikmin, ikmax;
  double rand, phi, theta;
  const double twopi = 8.e0 * std::atan(1.0);

  // the amplitude of the modes at kmin and kmax for a parabolic Fourier spectrum wrt 1.0 at the centre kc
  bool Debug = false;

  // initialize some variables, allocate random seed

  st_OUvar = std::sqrt(st_energy/st_decay);
  std::cout << st_energy << ' ' << st_decay << ' ' << st_OUvar << ' ' << 1.0/st_OUvar << std::endl;

  // this is for st_spectform = 1 (paraboloid) only
  // prefactor for amplitude normalistion to 1 at kc = 0.5*(st_stirmin+st_stirmax)
  parab_prefact = -4.0 / ((st_stirmax-st_stirmin)*(st_stirmax-st_stirmin));

  // characteristic k for scaling the amplitude below
  kc = st_stirmin;
  if (st_spectform == 1){ kc = 0.5*(st_stirmin+st_stirmax);}

  // this makes the rms force const irrespective of the solenoidal weight
  //if (ndim == 3){ st_solweightnorm = std::sqrt(3.0/3.0)*std::sqrt(3.0)*1.0/std::sqrt(1.0-2.0*st_solweight+3.0*st_solweight*st_solweight); }
  //if (ndim == 2){ st_solweightnorm = std::sqrt(3.0/2.0)*std::sqrt(3.0)*1.0/std::sqrt(1.0-2.0*st_solweight+2.0*st_solweight*st_solweight); }
  //if (ndim == 1){ st_solweightnorm = std::sqrt(3.0/1.0)*std::sqrt(3.0)*1.0/std::sqrt(1.0-2.0*st_solweight+1.0*st_solweight*st_solweight); }
  
  if (ndim == 3){ st_solweightnorm = std::sqrt(3.0)/std::sqrt(1.0-2.0*st_solweight+3.0*st_solweight*st_solweight); }
  if (ndim == 2){ st_solweightnorm = std::sqrt(0.5)*3.0/std::sqrt(1.0-2.0*st_solweight+2.0*st_solweight*st_solweight); }
  if (ndim == 1){ st_solweightnorm = 3.0/std::sqrt(1.0-2.0*st_solweight+1.0*st_solweight*st_solweight); }

  ikxmin = 0;
  ikymin = 0;
  ikzmin = 0;

  ikxmax = 256;
  ikymax = 0;
  ikzmax = 0;
  if (ndim > 1){ ikymax = 256; }
  if (ndim > 2){ ikzmax = 256; }

  // determine the number of required modes (in case of full sampling)
  st_nmodes = 0;
  for(ikx = ikxmin; ikx<=ikxmax; ikx++){
     kx = twopi * ikx / Lx;
     for(iky = ikymin; iky<=ikymax; iky++){
        ky = twopi * iky / Ly;
        for(ikz = ikzmin; ikz<=ikzmax; ikz++){
           kz = twopi * ikz / Lz;
           k = std::sqrt( kx*kx + ky*ky + kz*kz );
           if ((k >= st_stirmin) && (k <= st_stirmax)){
              st_nmodes = st_nmodes + 1;
              if (ndim > 1){ st_nmodes = st_nmodes + 1; }
              if (ndim > 2){ st_nmodes = st_nmodes + 2; }
           }
        }
     }
  }
  st_tot_nmodes = st_nmodes;
  if (st_spectform != 2){ std::cout << "Generating " << st_tot_nmodes << " driving modes..." << std::endl;}

  st_nmodes = -1;
  
  if (st_spectform != 2){
  // ===================================================================
  // === for band and parabolic spectrum, use the standard full sampling
      //open(13, file='power.txt', action='write')
      //write(13, '(6A16)') 'k', 'power', 'amplitude', 'kx', 'ky', 'kz'
      //close(13)

      // loop over all kx, ky, kz to generate driving modes
     for(ikx = ikxmin; ikx<=ikxmax; ikx++){
       kx = twopi * ikx / Lx;
       for(iky = ikymin; iky<=ikymax; iky++){
         ky = twopi * iky / Ly;
         for(ikz = ikzmin; ikz<=ikzmax; ikz++){
           kz = twopi * ikz / Lz;
           k = std::sqrt( kx*kx + ky*ky + kz*kz );
              
              if ((k >= st_stirmin) && (k <= st_stirmax)){
                 
                 if ((st_nmodes + std::pow(2,ndim-1)) > st_maxmodes){
                    
                    std::cout << "init_stir:  st_nmodes = " << st_nmodes << " maxstirmodes = " << st_maxmodes << std::endl;
                    std::cout << "Too many stirring modes" << std::endl;
                    break;                                                                       
                    
                 }
                 
                 if (st_spectform == 0){ amplitude = 1.0; }                               // Band
                 if (st_spectform == 1){ amplitude = std::abs(parab_prefact*(k-kc)*(k-kc)+1.0); } // Parabola
                 
                 // note: power spectrum ~ amplitude^2 (1D), amplitude^2 * 2pi k (2D), amplitude^2 * 4pi k^2 (3D)
                 amplitude = std::sqrt(amplitude) * std::pow((kc/k),0.5*(ndim-1));
                 
                 st_nmodes = st_nmodes + 1;
                 
                 st_ampl[st_nmodes] = amplitude;
                 //if (Debug) print *, "init_stir:  st_ampl(",st_nmodes,") = ", st_ampl(st_nmodes);        //HELP!
                 
                 st_mode[st_nmodes][0] = kx;
                 st_mode[st_nmodes][1] = ky;
                 st_mode[st_nmodes][2] = kz;
                 
                 if (ndim>1){
                    
                    st_nmodes = st_nmodes + 1;
                    
                    st_ampl[st_nmodes] = amplitude;
                    //if (Debug) print *, "init_stir:  st_ampl(",st_nmodes,") = ", st_ampl[st_nmodes];
                    
                    st_mode[st_nmodes][0] = kx;
                    st_mode[st_nmodes][1] =-ky;
                    st_mode[st_nmodes][2] = kz;
                    
                 }
                 
                 if (ndim>2) {
                    
                    st_nmodes = st_nmodes + 1;
                    
                    st_ampl[st_nmodes] = amplitude;
                    //if (Debug) print *, "init_stir:  st_ampl(",st_nmodes,") = ", st_ampl[st_nmodes];
                    
                    st_mode[st_nmodes][0] = kx;
                    st_mode[st_nmodes][1] = ky;
                    st_mode[st_nmodes][2] =-kz;
                    
                    st_nmodes = st_nmodes + 1;
                    
                    st_ampl[st_nmodes] = amplitude;
                    //if (Debug) print *, "init_stir:  st_ampl(",st_nmodes,") = ", st_ampl(st_nmodes);
                    
                    st_mode[st_nmodes][0] = kx;
                    st_mode[st_nmodes][1] =-ky;
                    st_mode[st_nmodes][2] =-kz;
                    
                 }
                 
                 //open(13, file='power.txt', action='write', access='append')
                 //write(13, '(6E16.6)') k, amplitude**2*(k/kc)**(ndim-1), amplitude, kx, ky, kz
                 //close(13)
                 
                 if (st_nmodes%1000 == 0){
                       //write(*,'(A,I6,A,I6,A)') ' ...', st_nmodes, ' of total ', st_tot_nmodes, ' modes generated...'}
                       std::cout << " ..." << st_nmodes << " of total " << st_tot_nmodes << " modes generated..." << std::endl;}
                 
              } // in k range
           } // ikz
        }// iky
     } // ikx
  }
  st_nmodes = st_nmodes + 1;
  return;

}
