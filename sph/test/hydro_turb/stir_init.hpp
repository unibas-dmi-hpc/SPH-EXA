#pragma once

#include <map>
#include <cmath>
#include <algorithm>
#include <vector>
#include <iostream>
using T = double;
void stir_init( std::vector<T>& stAmpl, std::vector<T>& stMode,size_t& stNModes, T& stSolWeight, T& stSolWeightNorm, T& stOUVar, T& stDecay, T Lx,T Ly,T Lz, size_t st_maxmodes,T st_energy,T st_stirmax,T st_stirmin,size_t ndim,size_t st_spectform){

  size_t ikxmin, ikxmax, ikymin, ikymax, ikzmin, ikzmax;
  size_t ikx, iky, ikz, st_tot_nmodes;
  T kx, ky, kz, k, kc, amplitude, parab_prefact;

  // for uniform random numbers in case of power law (st_spectform .eq. 2)
  size_t iang, nang, ik, ikmin, ikmax;
  T rand, phi, theta;
  const T twopi = 2.0 * M_PI;

  // the amplitude of the modes at kmin and kmax for a parabolic Fourier spectrum wrt 1.0 at the centre kc
  // initialize some variables, allocate random seed

  stOUVar = std::sqrt(st_energy / stDecay);

  // this is for st_spectform = 1 (paraboloid) only
  // prefactor for amplitude normalistion to 1 at kc = 0.5*(st_stirmin+st_stirmax)
  parab_prefact = -4.0 / ((st_stirmax - st_stirmin) * (st_stirmax - st_stirmin));

  // characteristic k for scaling the amplitude below
  kc = st_stirmin;
  if (st_spectform == 1){ kc = 0.5 * (st_stirmin + st_stirmax);}

  // this makes the rms force const irrespective of the solenoidal weight
  stSolWeightNorm = std::sqrt(3.0) * std::sqrt(3.0/T(ndim)) / std::sqrt(1.0 - 2.0 * stSolWeight + T(ndim) * stSolWeight * stSolWeight);

  ikxmin = 0;
  ikymin = 0;
  ikzmin = 0;

  ikxmax = 256;
  ikymax = 0;
  ikzmax = 0;
  if (ndim > 1){ ikymax = 256; }
  if (ndim > 2){ ikzmax = 256; }

  // determine the number of required modes (in case of full sampling)
  stNModes = 0;
  for(ikx = ikxmin; ikx <= ikxmax; ikx++){
     kx = twopi * ikx / Lx;
     for(iky = ikymin; iky <= ikymax; iky++){
        ky = twopi * iky / Ly;
        for(ikz = ikzmin; ikz <= ikzmax; ikz++){
           kz = twopi * ikz / Lz;
           k = std::sqrt( kx * kx + ky * ky + kz * kz );
           if ((k >= st_stirmin) && (k <= st_stirmax)){
              stNModes += 1;
              if (ndim > 1){ stNModes += 1; }
              if (ndim > 2){ stNModes += 2; }
           }
        }
     }
  }
  st_tot_nmodes = stNModes;
  if (st_spectform != 2){ std::cout << "Generating " << st_tot_nmodes << " driving modes..." << std::endl;}

  stNModes = -1;

  if (st_spectform != 2){
  // ===================================================================
  // === for band and parabolic spectrum, use the standard full sampling
      //open(13, file='power.txt', action='write')
      //write(13, '(6A16)') 'k', 'power', 'amplitude', 'kx', 'ky', 'kz'
      //close(13)

      // loop over all kx, ky, kz to generate driving modes
     for(ikx = ikxmin; ikx <= ikxmax; ikx++){
       kx = twopi * ikx / Lx;
       for(iky = ikymin; iky <= ikymax; iky++){
         ky = twopi * iky / Ly;
         for(ikz = ikzmin; ikz <= ikzmax; ikz++){
           kz = twopi * ikz / Lz;
           k = std::sqrt( kx * kx + ky * ky + kz * kz );

              if ((k >= st_stirmin) && (k <= st_stirmax)){

                 if ((stNModes + 1 + std::pow(2,ndim-1)) > st_maxmodes){

                    std::cout << "init_stir:  number of modes: = " << stNModes+1 << " maxstirmodes = " << st_maxmodes << std::endl;
                    std::cout << "Too many stirring modes" << std::endl;
                    break;

                 }

                 if (st_spectform == 0){ amplitude = 1.0; }                               // Band
                 if (st_spectform == 1){ amplitude = std::abs(parab_prefact * (k - kc) * (k - kc) + 1.0); } // Parabola

                 // note: power spectrum ~ amplitude^2 (1D), amplitude^2 * 2pi k (2D), amplitude^2 * 4pi k^2 (3D)
                 amplitude  = 2.0 * std::sqrt(amplitude) * std::pow((kc / k), 0.5 * (ndim - 1));

                 stNModes += 1;

                 stAmpl[stNModes] = amplitude;
                 //if (Debug) print *, "init_stir:  stAmpl(",stNModes,") = ", stAmpl(stNModes);        //HELP!

                 stMode[ndim * stNModes]     = kx;
                 stMode[ndim * stNModes + 1] = ky;
                 stMode[ndim * stNModes + 2] = kz;

                 if (ndim>1){

                    stNModes += 1;

                    stAmpl[stNModes] = amplitude;
                    //if (Debug) print *, "init_stir:  stAmpl(",stNModes,") = ", stAmpl[stNModes];

                    stMode[ndim * stNModes]     =  kx;
                    stMode[ndim * stNModes + 1] = -ky;
                    stMode[ndim * stNModes + 2] =  kz;

                 }

                 if (ndim>2) {

                    stNModes += 1;

                    stAmpl[stNModes] = amplitude;
                    //if (Debug) print *, "init_stir:  stAmpl(",stNModes,") = ", stAmpl[stNModes];

                    stMode[ndim * stNModes]     =  kx;
                    stMode[ndim * stNModes + 1] =  ky;
                    stMode[ndim * stNModes + 2] = -kz;

                    stNModes += 1;

                    stAmpl[stNModes] = amplitude;

                    stMode[ndim*stNModes] = kx;
                    stMode[ndim*stNModes+1] = -ky;
                    stMode[ndim*stNModes+2] = -kz;
                 }

                 if (stNModes %  1000 == 0){
                       std::cout << " ..." << stNModes << " of total " << st_tot_nmodes << " modes generated..." << std::endl;}
              } // in k range
           } // ikz
        }// iky
     } // ikx
  }
  stNModes += 1;
  return;
}
