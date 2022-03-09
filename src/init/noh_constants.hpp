#pragma once

namespace sphexa
{

struct NohConstants
{
    static inline const double   r0            = 0.;
    static inline const double   r1            = 0.5;
    static inline const double   mTotal        = 1.;
    static inline const unsigned dim           = 3;
    static inline const double   gamma         = 5. / 3.;
    static inline const double   rho0          = 1.;
    static inline const double   u0            = 1.e-20;
    static inline const double   p0            = 0.;
    static inline const double   vr0           = -1.;
    static inline const double   cs0           = 0.;
    static inline const double   firstTimeStep = 1.e-4;
};

} // namespace sphexa
