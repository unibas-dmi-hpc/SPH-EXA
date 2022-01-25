#pragma once

#include <cmath>
#include <vector>

#include "sph/kernels.hpp"
#include "ParticlesData.hpp"

namespace sphexa
{
template <typename T, typename I>
class SedovDataGenerator
{
public:

    static inline const double gamma         = 5.0/3.0;
    static inline const double r0            = 0.;
    static inline const double r1            = 0.5;
    static inline const double energytot     = 1.0;
    static inline const double width         = 0.10;
    static inline const double rho0          = 1.0;
    static inline const double ener0         = energytot / std::pow(M_PI,(3.0/2.0)) / 1.0 / std::pow(width,3);
    static inline const T      firstTimeStep = 1e-6;

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

        pd.n = side * side * side;
        pd.side = side;
        pd.count = side * side * side;

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
                      << pd.count * (pd.data.size() * 64.) / (8. * 1000. * 1000.0 * 1000.0)
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

                        pd.vx[lindex - offset] = 0.0;
                        pd.vy[lindex - offset] = 0.0;
                        pd.vz[lindex - offset] = 0.0;
                    }
                }
            }
        }
    }

    static void init(ParticlesData<T, I> &pd)
    {
        const T dx = 1.0 / pd.side;

        #pragma omp parallel for
        for (size_t i = 0; i < pd.count; i++)
        {
            double radius = sqrt(pd.x[i] * pd.x[i] +  pd.y[i] * pd.y[i]+ pd.z[i] * pd.z[i]);

            pd.u[i] = ener0 * exp(-(std::pow(radius,2) / std::pow(width,2))) + 1.e-08;

            pd.p[i] = pd.u[i]*1.0*(gamma-1.0);

            pd.m[i] = 1.0 / pd.n; // 1.0;//1000000.0/n;//1.0;//0.001;//0.001;//0.001;//1.0;
            //pd.c[i] = 3500.0;           // 35.0;//35.0;//35000
            pd.h[i] = 1.5 * dx;//0.28577500E-01 / 2.0; //2.0 * dx;         // 0.02;//0.02;
            pd.ro[i] = rho0;  // 1.0 // 1.0e3;//.0;//1e3;//1e3;

            pd.mui[i] = 10.0;

            pd.du[i] = pd.du_m1[i] = 0.0;
            pd.dt[i] = pd.dt_m1[i] = firstTimeStep;
            pd.minDt = firstTimeStep;

            pd.grad_P_x[i] = pd.grad_P_y[i] = pd.grad_P_z[i] = 0.0;

            pd.x_m1[i] = pd.x[i] - pd.vx[i] * firstTimeStep;
            pd.y_m1[i] = pd.y[i] - pd.vy[i] * firstTimeStep;
            pd.z_m1[i] = pd.z[i] - pd.vz[i] * firstTimeStep;
        }

        pd.etot = pd.ecin = pd.eint = 0.0;
        pd.ttot = 0.0;
    }
};

} // namespace sphexa
