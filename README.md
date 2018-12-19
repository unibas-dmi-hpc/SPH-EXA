# SPH-EXA mini-app
The SPH-EXA mini-app implements the smoothed particle hydrodynamics (SPH) technique, a meshless Lagrangian method commonly used for performing hydrodynamical and computational fluid dynamics simulations.

The SPH technique discretizes a fluid in a series of interpolation points (SPH particles) whose distribution follows
the mass density of the fluid and their evolution relies on a weighted interpolation over close neighboring particles.
SPH simulations with detailed physics calculations represent computationally-demanding applications. 
The SPH-EXA mini-app is derived from three parent SPH codes used in astrophysics (SPHYNX and ChaNGa) and computational fluid dynamics (SPH-flow).

A number of basic steps of any SPH calculation are included in the mini-app: from the particles’ positions and masses a tree is built and walked to identify the neighbors that will be used for the remainder of the global time-step (or iteration). Such steps include the evaluation of the particles’ density, acceleration, rate of change of internal energy, and all physical modules relevant to the studied scenario. Next, a new physically relevant and numerically stable time-step is found, and the properties of the particles are updated accordingly.

SPH-EXA mini-app is a modern C++ headers-only code (except for main.cpp) with no external software dependencies.
The parallelism is currently expressed via MPI+OpenMP and will be extended to exploit accelerated parallelism (OpenACC, HPX).

## Test cases
### 3D Rotating Square Patch
This mini-app can simulate a three-dimensional rotating square patch, a demanding scenario for SPH simulations due to the presence of negative pressures, which stimulate the emergence of unphysical tensile instabilities that destroy the particle system, unless corrective repulsive forces are included.

### Evrard Collapse
A second demanding simulation scenario carried out by the SPH-EXA mini-app is the Evrard collapse: it consists of an initially static isothermal, spherical cloud of gas (mimicking a star) that undergoes an accelerated gravitational collapse, until the rapid rise of temperature and pressure at its core produces a shock wave that expands from the center of the star to its outer layers. The Evrard collapse involves ingredients that are crucial for astrophysical simulations, namely shock waves and self-gravity.

### Blob test
The mini-app can also simulate a wind-cloud scenario (also known as Blob test). This problem reunites several types of physics, such as strong shocks and mixing due to hydrodynamical instabilities in a multiphase medium with a large density contrast. The initial configuration consists of a dense spherical cloud of cold gas embedded in a hotter ambient medium. The cloud is initially at rest while the ambient background (the wind) moves supersonically, ablating and finally destroying the bubble. 
