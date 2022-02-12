#pragma once

#include <iostream>
#include <cmath>
#include <vector>

#include "sph/kernels.hpp"
#include "particles_data.hpp"

namespace sphexa
{
template <typename T, typename I>
class SedovDataGenerator
{
public:

    static inline const I dim           = 3;
    static inline const T gamma         = 5./3.;
    static inline const T omega         = 0.;
    static inline const T r0            = 0.;
    static inline const T r1            = 0.5;
    static inline const T mTotal        = 1.;
    static inline const T energyTotal   = 1.;
    static inline const T width         = 0.1;
    static inline const T ener0         = energyTotal / std::pow(M_PI, 1.5) / (width * width * width);
    static inline const T rho0          = 1.;
    static inline const T u0            = 1.e-08;
    static inline const T p0            = 0.;
    static inline const T vel0          = 0.;
    static inline const T cs0           = 0.;
    static inline const T firstTimeStep = 1.e-6;

    static ParticlesData<T, I> generate(const size_t side)
    {
        ParticlesData<T, I> pd;

        if (pd.rank == 0 && side < 8)
        {
            printf("ERROR::Sedov::init()::SmoothingLength n too small\n");
            #ifdef USE_MPI
            MPI_Finalize();
            #endif
            exit(0);
        }

        #ifdef USE_MPI
        pd.comm = MPI_COMM_WORLD;
        MPI_Comm_size(pd.comm, &pd.nrank);
        MPI_Comm_rank(pd.comm, &pd.rank);
        MPI_Get_processor_name(pd.pname, &pd.pnamelen);
        #endif

        pd.side  = side;
        pd.n     = side * side * side;
        pd.count = pd.n;

        load(pd);
        init(pd);

        return pd;
    }

    // void load(const std::string &filename)
    static void load(ParticlesData<T, I> &pd)
    {
        size_t split = pd.n / pd.nrank;
        size_t remaining = pd.n - pd.nrank * split;

        pd.count = split;
        if (pd.rank == 0) pd.count += remaining;

        pd.resize(pd.count);

        if(pd.rank == 0)
            std::cout << "Approx: "
                      << pd.count * (pd.data.size() * 64.) / (8. * 1000. * 1000. * 1000.)
                      << "GB allocated on rank 0."
                      << std::endl;

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

                        pd.z[ lindex - offset] = lz;
                        pd.y[ lindex - offset] = ly;
                        pd.x[ lindex - offset] = lx;

                        pd.vx[lindex - offset] = vel0;
                        pd.vy[lindex - offset] = vel0;
                        pd.vz[lindex - offset] = vel0;
                    }
                }
            }
        }
    }

    static void init(ParticlesData<T, I> &pd)
    {
        const T step  = (2. * r1) / pd.side;    //
        const T hIni  = 1.5 * step;             //
        const T mPart = mTotal / pd.n;          //
        const T gamm1 = gamma - 1.;             //

        #pragma omp parallel for
        for (size_t i = 0; i < pd.count; i++)
        {
            const T radius = std::sqrt( (pd.x[i] * pd.x[i]) + (pd.y[i] * pd.y[i]) + (pd.z[i] * pd.z[i]) );

            pd.h[i]        = hIni;
            pd.m[i]        = mPart;
            pd.ro[i]       = rho0;
            pd.u[i]        = ener0 * exp( -(radius * radius) / (width * width) ) + u0;
            pd.p[i]        = pd.u[i] * rho0 * gamm1;

            pd.mui[i]      = 10.;

            pd.du[i]       = 0.;
            pd.du_m1[i]    = 0.;

            pd.dt[i]       = firstTimeStep;
            pd.dt_m1[i]    = firstTimeStep;
            pd.minDt       = firstTimeStep;

            pd.grad_P_x[i] = 0.;
            pd.grad_P_y[i] = 0.;
            pd.grad_P_z[i] = 0.;

            pd.x_m1[i]     = pd.x[i] - pd.vx[i] * firstTimeStep;
            pd.y_m1[i]     = pd.y[i] - pd.vy[i] * firstTimeStep;
            pd.z_m1[i]     = pd.z[i] - pd.vz[i] * firstTimeStep;
        }

        pd.etot = 0.;
        pd.ecin = 0.;
        pd.eint = 0.;
        pd.ttot = 0.;
    }
};

} // namespace sphexa
