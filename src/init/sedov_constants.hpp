#pragma once

namespace sphexa
{

struct SedovConstants
{
    static inline const unsigned dim           = 3;
    static inline const double   gamma         = 5. / 3.;
    static inline const double   omega         = 0.;
    static inline const double   r0            = 0.;
    static inline const double   r1            = 0.5;
    static inline const double   mTotal        = 1.;
    static inline const double   energyTotal   = 1.;
    static inline const double   width         = 0.1;
    static inline const double   ener0         = energyTotal / std::pow(M_PI, 1.5) / 1. / std::pow(width, 3.0);
    static inline const double   rho0          = 1.;
    static inline const double   u0            = 1.e-08;
    static inline const double   p0            = 0.;
    static inline const double   vr0           = 0.;
    static inline const double   cs0           = 0.;
    static inline const double   firstTimeStep = 1.e-6;
};

} // namespace sphexa
