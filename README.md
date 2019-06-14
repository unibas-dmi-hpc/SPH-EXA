# SPH-EXA mini-app
The SPH-EXA mini-app implements the smoothed particle hydrodynamics (SPH) technique, a meshless Lagrangian method commonly used for performing hydrodynamical and computational fluid dynamics simulations.

The SPH technique discretizes a fluid in a series of interpolation points (SPH particles) whose distribution follows
the mass density of the fluid and their evolution relies on a weighted interpolation over close neighboring particles.
SPH simulations with detailed physics calculations represent computationally-demanding applications. 
The SPH-EXA mini-app is derived from three parent SPH codes used in astrophysics (SPHYNX and ChaNGa) and computational fluid dynamics (SPH-flow).

A number of basic steps of any SPH calculation are included in the mini-app: from the particles’ positions and masses a tree is built and walked to identify the neighbors that will be used for the remainder of the global time-step (or iteration). Such steps include the evaluation of the particles’ density, acceleration, rate of change of internal energy, and all physical modules relevant to the studied scenario. Next, a new physically relevant and numerically stable time-step is found, and the properties of the particles are updated accordingly.

SPH-EXA mini-app is a modern C++ headers-only code (except for main.cpp) with no external software dependencies.
The parallelism is currently expressed via MPI+OpenMP and will be extended to exploit accelerated parallelism (OpenACC, HPX).

### 3D Rotating Square Patch
This mini-app can simulate a three-dimensional rotating square patch, a demanding scenario for SPH simulations due to the presence of negative pressures, which stimulate the emergence of unphysical tensile instabilities that destroy the particle system, unless corrective repulsive forces are included.
