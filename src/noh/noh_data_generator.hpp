#pragma once

#include <iostream>
#include <cmath>
#include <vector>

#include "particles_data.hpp"

namespace sphexa
{

class NohDataGenerator
{
public:
    static inline const unsigned dim           = 3;
    static inline const double   gamma         = 5. / 3.;
    static inline const double   r0            = 0.;
    static inline const double   r1            = 0.5;
    static inline const double   mTotal        = 1.;
    static inline const double   vr0           = -1.;
    static inline const double   rho0          = 1.;
    static inline const double   u0            = 1.e-20;
    static inline const double   p0            = 0.;
    static inline const double   vel0          = 0.;
    static inline const double   cs0           = 0.;
    static inline const double   firstTimeStep = 1.e-4;

    template<class Dataset>
    static Dataset generate(const size_t side)
    {
        Dataset pd;

        if (pd.rank == 0 && side < 8)
        {
            printf("ERROR::Noh::init()::SmoothingLength n too small\n");
#ifdef USE_MPI
            MPI_Finalize();
#endif
            exit(0);
        }

#ifdef USE_MPI
        pd.comm = MPI_COMM_WORLD;
        MPI_Comm_size(pd.comm, &pd.nrank);
        MPI_Comm_rank(pd.comm, &pd.rank);
#endif

        pd.side  = side;
        pd.n     = side * side * side;
        pd.count = pd.n;

        load(pd);
        init(pd);

        return pd;
    }

    template<class Dataset>
    static void load(Dataset& pd)
    {
        size_t split     = pd.n / pd.nrank;
        size_t remaining = pd.n - pd.nrank * split;

        pd.count = split;
        if (pd.rank == 0) pd.count += remaining;

        resize(pd, pd.count);

        if (pd.rank == 0)
            std::cout << "Approx: " << pd.count * (pd.data().size() * 64.) / (8. * 1000. * 1000. * 1000.)
                      << "GB allocated on rank 0." << std::endl;

        size_t offset = pd.rank * split;
        if (pd.rank > 0) offset += remaining;

        double step = (2. * r1) / pd.side;

#pragma omp parallel for
        for (size_t i = 0; i < pd.side; ++i)
        {
            double lz = -r1 + (i * step);

            for (size_t j = 0; j < pd.side; ++j)
            {
                double lx = -r1 + (j * step);

                for (size_t k = 0; k < pd.side; ++k)
                {
                    size_t lindex = (i * pd.side * pd.side) + (j * pd.side) + k;

                    if (lindex >= offset && lindex < offset + pd.count)
                    {
                        double ly = -r1 + (k * step);

                        pd.z[lindex - offset] = lz;
                        pd.y[lindex - offset] = ly;
                        pd.x[lindex - offset] = lx;

                        pd.vx[lindex - offset] = vel0;
                        pd.vy[lindex - offset] = vel0;
                        pd.vz[lindex - offset] = vel0;
                    }
                }
            }
        }
    }

    template<class Dataset>
    static void init(Dataset& pd)
    {
        const double step  = (2. * r1) / pd.side;
        const double hIni  = 1.5 * step;
        const double mPart = mTotal / pd.n;

#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < pd.count; i++)
        {
            const double radius = std::sqrt((pd.x[i] * pd.x[i]) + (pd.y[i] * pd.y[i]) + (pd.z[i] * pd.z[i]));

            pd.h[i] = hIni;
            pd.m[i] = mPart;
            pd.u[i] = u0;

            pd.vx[i] = vr0 * (pd.x[i] / radius);
            pd.vy[i] = vr0 * (pd.y[i] / radius);
            pd.vz[i] = vr0 * (pd.z[i] / radius);

            // pd.mui[i] = 10.;
            pd.du[i]    = 0.;
            pd.du_m1[i] = 0.;

            pd.dt[i]    = firstTimeStep;
            pd.dt_m1[i] = firstTimeStep;
            pd.minDt    = firstTimeStep;

            pd.x_m1[i] = pd.x[i] - pd.vx[i] * firstTimeStep;
            pd.y_m1[i] = pd.y[i] - pd.vy[i] * firstTimeStep;
            pd.z_m1[i] = pd.z[i] - pd.vz[i] * firstTimeStep;
        }

        pd.etot = 0.;
        pd.ecin = 0.;
        pd.eint = 0.;
        pd.ttot = 0.;
    }
};

} // namespace sphexa
