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
 * @brief turbulence_module: computes some constants and declares variables
 * @author Axel Sanz <axel.sanz@estudiantat.upc.edu>
 */
  const int st_maxmodes = 100000;
  const double twopi = 8.0e0 * std::atan(1.0);

  // OU variance corresponding to decay time and energy input rate
  double st_OUvar;

  // number of modes
  int st_nmodes, st_seed;

  double st_mode[st_maxmodes][3], st_aka[st_maxmodes][3], st_akb[st_maxmodes][3];
  double st_OUphases[6*st_maxmodes];
  double st_ampl[st_maxmodes];

  const int     ndim=dim; // number of spatial dimensions

  const double  st_solweight = 0.5e0;
  double        st_solweightnorm;
  const double  st_power_law_exp = 1.5e0;
  const double  st_angles_exp = 1.e0;
  const double  eps = 1.0e-16;  //HELP (Buscar algo parecido o sino 1.0e-16) (https://en.cppreference.com/w/cpp/types/numeric_limits/epsilon)
  const double  Lbox = std::min(std::min(Lx,Ly),Lz), velocity = 0.3e0;
  const double  st_decay = Lbox/(2.0*velocity);
  const double  st_energy = 5.0e-3 * pow(velocity,3)/Lbox;
  const double  st_stirmin = (1.e0-eps) * twopi/Lbox, st_stirmax = (3.e0+eps) * twopi/Lbox;

  //int st_seed_ini = 140281;
  int st_seed_ini = 251299;
  int st_spectform = 1;



//  ndim = dim;                                // 3-dimensional forcing

//  xmin = -0.5; xmax = 0.5;                   // Spatial x-coordinates of the box (xmax-xmin=L_box)
//  ymin = -0.5; ymax = 0.5;                   // Spatial y-coordinates of the box (ymax-ymin=L_box)
//  zmin = -0.5; zmax = 0.5;                   // Spatial z-coordinates of the box (zmax-zmin=L_box)
//  L = xmax-xmin
                                             // Spectral shape of the driving field
//  st_spectform     = 1                       // 0 is band, 1 is paraboloid, 2: power law
//  st_power_law_exp = 1.5                     // st_spectform = 2: spectral power-law exponent
//  st_angles_exp    = 1.0                     // st_spectform = 2: number of modes (angles) in k-shell surface increases as k^st_angles_exp
                                             // for full sampling, st_angles_exp = 2.0; for healpix-type sampling, st_angles_exp = 0.0

//  velocity     = 1.0                         // Target velocity dispersion

//  st_decay     = L/(2.0*velocity)            // Auto-correlation time, T=L_box/(2V); aka turbulent turnover (crossing) time

//  st_energy    = 5.0d-3 * velocity**3/L      // Energy input rate => driving amplitude ~ sqrt(energy/decay)
                                             // Note that energy input rate ~ velocity^3 * L_box^-1
                                             // Pre-factor needs to be adjusted to approach actual target velocity dispersion

//  st_stirmin   = (1.e0-eps) * twopi/L           // <~  1 * 2pi / L_box ; k=1.0
//  st_stirmax   = (3.e0+eps) * twopi/L           // >~  3 * 2pi / L_box ; k=3.0
                                             // Minimum and maximum wavenumbers for driving

//  st_solweight = 1                          // 1.0 solenoidal; 0.0 compressive; 0.5 natural mixture
                                             // See Federrath et al. (2010, A&A 512, A81) for details

//  st_seed    = 140281                        // Random seed
