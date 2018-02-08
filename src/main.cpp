#include <iostream>

#include "Simulation.hpp"

int main(int argc, char** argv) 
{
  int N;			//number of particles
  int nsteps; 		//number ot integration steps
  
  Simulation sim;
    
  if(argc>1)
  {
    N=atoi(argv[1]);
    sim.set_number_of_particles(N);  
    if(argc==3) 
    {
      nsteps=atoi(argv[2]);
      sim.set_number_of_steps(nsteps);  
    }
  }
  
  sim.start();

  return 0;
}