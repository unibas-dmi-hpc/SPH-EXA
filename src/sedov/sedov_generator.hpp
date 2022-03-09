#pragma once

#include <iostream>
#include <cmath>
#include <vector>

#include "init/grid.hpp"

namespace sphexa
{

class SedovDataGenerator
{
public:
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

    template<class Dataset>
    static void generate(Dataset& pd)
    {
        pd.n = pd.side * pd.side * pd.side;

        auto [first, last] = partitionRange(pd.n, pd.rank, pd.nrank);
        pd.count           = last - first;

        resize(pd, pd.count);

        if (pd.rank == 0)
        {
            std::cout << "Approx: " << pd.count * (pd.data().size() * 64.) / (8. * 1000. * 1000. * 1000.)
                      << "GB allocated on rank 0." << std::endl;
        }

        regularGrid(r1, pd.side, first, last, pd.x, pd.y, pd.z);
        init(pd);
    }

    template<class Dataset>
    static void init(Dataset& pd)
    {
        using T = typename Dataset::RealType;

        const double step   = (2. * r1) / pd.side;
        const double hIni   = 1.5 * step;
        const double mPart  = mTotal / pd.n;
        const double width2 = width * width;
        std::fill(pd.m.begin(), pd.m.end(), mPart);
        std::fill(pd.h.begin(), pd.h.end(), hIni);
        std::fill(pd.du_m1.begin(), pd.du_m1.end(), 0.0);
        std::fill(pd.mui.begin(), pd.mui.end(), 10.0);
        std::fill(pd.dt.begin(), pd.dt.end(), firstTimeStep);
        std::fill(pd.dt_m1.begin(), pd.dt_m1.end(), firstTimeStep);
        std::fill(pd.alpha.begin(), pd.alpha.end(), pd.alphamin);
        pd.minDt = firstTimeStep;

        std::fill(pd.vx.begin(), pd.vx.end(), 0.0);
        std::fill(pd.vy.begin(), pd.vy.end(), 0.0);
        std::fill(pd.vz.begin(), pd.vz.end(), 0.0);

#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < pd.count; i++)
        {
            T xi = pd.x[i];
            T yi = pd.y[i];
            T zi = pd.z[i];
            T r2 = xi * xi + yi * yi + zi * zi;

            pd.u[i] = ener0 * exp(-(r2 / width2)) + u0;

            pd.x_m1[i] = xi - pd.vx[i] * firstTimeStep;
            pd.y_m1[i] = yi - pd.vy[i] * firstTimeStep;
            pd.z_m1[i] = zi - pd.vz[i] * firstTimeStep;
        }
    }
};

} // namespace sphexa
