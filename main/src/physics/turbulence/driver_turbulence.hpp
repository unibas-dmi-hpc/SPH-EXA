#pragma once

#include <cmath>
#include <iostream>
#include "st_ounoise.hpp"
#include "st_calcAccel.hpp"
#include "st_calcPhases.hpp"

namespace sphexa{

template<class Dataset, class T>
void driver_turbulence(size_t first,size_t last,Dataset d){

  std::vector<T> st_aka(d.ndim*d.stNModes);
  std::vector<T> st_akb(d.ndim*d.stNModes);
  std::cout << "ounoiseupdate: " << std::endl;
  st_ounoiseupdate(d.stOUPhases, 6*d.stNModes, d.stOUvar, d.minDt, d.stDecay,d.stSeed);
  std::cout << "calc_phases: " << std::endl;
  st_calcPhases(d.stNModes,d.ndim,d.stOUPhases,d.stSolWeight,d.stMode,st_aka,st_akb);
  std::cout << "calc_accel: " << std::endl;
  st_calcAccel(first,last,d.x,d.y,d.z,d.ax,d.ay,d.az,d.stNModes,d.stMode,st_aka,st_akb,d.stAmpl,d.stSolWeightNorm);
}

} // namespace sphexa
