#include <cmath>
#include <iostream>
#include "st_ounoise.hpp"
#include "st_calcAccel.hpp"
#include "st_calcPhases.hpp"
#include "stir_init.hpp"

//st_seed
//st_OUvar
//st_mode                  // computed once in first iteration
//st_nmodes
//st_solweightnorm

//st_OUphases
void driver_turbulence(double dt,int npart,double st_OUphases[], int st_nmodes, double st_OUvar, double st_decay, int st_seed,
                       double st_solweight, double st_solweightnorm, double st_ampl,
                       st_mode[][3],double xCoord[],double yCoord[],double zCoord[],double accx[],double accy[],double accz[]){


  //T dt=d.minDt;
  //size_t dim=3;
  //size_t npart=d.x.size();

  double st_aka[st_maxmodes][3], st_akb[st_maxmodes][3];
  //double st_OUphases[6*st_maxmodes];
  //double st_ampl[st_maxmodes];

  const double  st_solweight = 0.5e0;
  double        st_solweightnorm;
  const double  eps = 1.0e-16;  //HELP (Buscar algo parecido o sino 1.0e-16) (https://en.cppreference.com/w/cpp/types/numeric_limits/epsilon)
  const double  Lbox = std::max(std::max(Lx,Ly),Lz), velocity = 0.3e0;
  const double  st_decay = Lbox/(2.0*velocity);
  const double  st_energy = 5.0e-3 * pow(velocity,3)/Lbox;
  const double  st_stirmin = (1.e0-eps) * twopi/Lbox, st_stirmax = (3.e0+eps) * twopi/Lbox;

  //int st_seed_ini = 140281;
  int st_seed_ini = 251299;
  int st_spectform = 1;

  st_ounoiseupdate(st_OUphases, 6*st_nmodes, st_OUvar, dt, st_decay,st_seed);
  st_calcPhases(st_nmodes,ndim,st_OUphases,st_solweight,st_mode,st_aka,st_akb);
  st_calcAccel(npart,xCoord,yCoord,zCoord,accx,accy,accz,st_nmodes,st_mode,st_aka,st_akb,st_ampl,st_solweightnorm);

}
