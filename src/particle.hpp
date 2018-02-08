#include <cmath>
#include "types.hpp"


struct Particle
{
  public:
    Particle() { init();}
    void init() 
    {
      pos_x = NULL; pos_y = NULL; pos_z = NULL;
      vel_x = NULL; vel_y = NULL; vel_z = NULL;
      acc_x = NULL; acc_y = NULL; acc_z = NULL;
      mass  = NULL;
    }
    real_type *pos_x, *pos_y, *pos_z;
    real_type *vel_x, *vel_y, *vel_z;
    real_type *acc_x, *acc_y, *acc_z;  
    real_type *mass;
};