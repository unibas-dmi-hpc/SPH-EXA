#pragma once

#include <cmath>
#include <vector>

#include "BBox.hpp"
#include "sph/kernels.hpp"
#include "ParticlesData.hpp"

namespace sphexa
{
template <typename T>
class SqPatchDataGenerator
{
public:
    static ParticlesData<T> generate(const size_t side)
    {
        ParticlesData<T> pd;

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
    static void load(ParticlesData<T> &pd)
    {
        pd.count = pd.n / pd.nrank;
        int offset = pd.n % pd.nrank;

        pd.workload.resize(pd.nrank);
        pd.displs.resize(pd.nrank);

        pd.workload[0] = pd.count + offset;
        pd.displs[0] = 0;

        for (int i = 1; i < pd.nrank; i++)
        {
            pd.workload[i] = pd.count;
            pd.displs[i] = pd.displs[i - 1] + pd.workload[i - 1];
        }

        pd.count = pd.workload[pd.rank];

        pd.resize(pd.count);

        const double omega = 5.0;
        const double myPI = std::acos(-1.0);

#pragma omp parallel for
        for (int i = 0; i < pd.side; ++i)
        {
            double lz = -0.5 + 1.0 / (2.0 * pd.side) + i * 1.0 / pd.side;

            for (int j = 0; j < pd.side; ++j)
            {
                // double ly = -0.5 + 1.0 / (2.0 * pd.side) +  (double)j / (double)pd.side;
                double lx = -0.5 + 1.0 / (2.0 * pd.side) + j * 1.0 / pd.side;

                for (int k = 0; k < pd.side; ++k)
                {
                    int lindex = i * pd.side * pd.side + j * pd.side + k;

                    if (lindex >= pd.displs[pd.rank] && lindex < pd.displs[pd.rank] + pd.workload[pd.rank])
                    {
                        double ly = -0.5 + 1.0 / (2.0 * pd.side) + k * 1.0 / pd.side;
                        // double lx = -0.5 + 1.0 / (2.0 * pd.side) + (double)k / (double)pd.side;

                        double lvx = omega * ly;
                        double lvy = -omega * lx;
                        double lvz = 0.;
                        double lp_0 = 0.;

                        for (int m = 1; m <= 39; m += 2)
                            for (int l = 1; l <= 39; l += 2)
                                lp_0 = lp_0 - 32.0 * (omega * omega) / (m * l * (myPI * myPI)) /
                                                  ((m * myPI) * (m * myPI) + (l * myPI) * (l * myPI)) * sin(m * myPI * (lx + 0.5)) *
                                                  sin(l * myPI * (ly + 0.5));

                        lp_0 *= 1000.0;

                        pd.z[lindex - pd.displs[pd.rank]] = lz;
                        pd.y[lindex - pd.displs[pd.rank]] = ly;
                        pd.x[lindex - pd.displs[pd.rank]] = lx;
                        pd.vx[lindex - pd.displs[pd.rank]] = lvx;
                        pd.vy[lindex - pd.displs[pd.rank]] = lvy;
                        pd.vz[lindex - pd.displs[pd.rank]] = lvz;
                        pd.p_0[lindex - pd.displs[pd.rank]] = lp_0;
                    }
                }
            }
        }
    }

    static void init(ParticlesData<T> &pd)
    {
        pd.dx = 100.0 / pd.side;

        for (int i = 0; i < pd.count; i++)
        {
            // CGS
            pd.x[i] = pd.x[i] * 100.0;
            pd.y[i] = pd.y[i] * 100.0;
            pd.z[i] = pd.z[i] * 100.0;
            pd.vx[i] = pd.vx[i] * 100.0;
            pd.vy[i] = pd.vy[i] * 100.0;
            pd.vz[i] = pd.vz[i] * 100.0;
            pd.p[i] = pd.p_0[i] = pd.p_0[i] * 10.0;

            pd.m[i] = 1000000.0 / pd.n; // 1.0;//1000000.0/n;//1.0;//0.001;//0.001;//0.001;//1.0;
            pd.c[i] = 3500.0;           // 35.0;//35.0;//35000
            pd.h[i] = 2.5 * pd.dx;      // 0.02;//0.02;
            pd.ro[i] = 1.0;             // 1.0e3;//.0;//1e3;//1e3;
            pd.ro_0[i] = 1.0;           // 1.0e3;//.0;//1e3;//1e3;

            pd.du[i] = pd.du_m1[i] = 0.0;
            pd.dt[i] = pd.dt_m1[i] = 1e-6;

            pd.grad_P_x[i] = pd.grad_P_y[i] = pd.grad_P_z[i] = 0.0;

            pd.x_m1[i] = pd.x[i] - pd.vx[i] * pd.dt[0];
            pd.y_m1[i] = pd.y[i] - pd.vy[i] * pd.dt[0];
            pd.z_m1[i] = pd.z[i] - pd.vz[i] * pd.dt[0];
        }

#ifdef USE_MPI
        pd.bbox.computeGlobal(pd.x, pd.y, pd.z, pd.comm);
#else
        pd.bbox.compute(pd.x, pd.y, pd.z);
#endif
        pd.bbox.PBCz = true;
        pd.bbox.zmax += pd.dx / 2.0;
        pd.bbox.zmin -= pd.dx / 2.0;

        pd.etot = pd.ecin = pd.eint = 0.0;
        pd.ttot = 0.0;

        if (pd.rank == 0 && 2.0 * pd.h[0] > (pd.bbox.zmax - pd.bbox.zmin) / 2.0)
        {
            printf("ERROR::SqPatch::init()::SmoothingLength (%.2f) too large (%.2f) (n too small?)\n", pd.h[0],
                   pd.bbox.zmax - pd.bbox.zmin);
#ifdef USE_MPI
            MPI_Finalize();
            exit(0);
#endif
        }
    }
};
} // namespace sphexa
