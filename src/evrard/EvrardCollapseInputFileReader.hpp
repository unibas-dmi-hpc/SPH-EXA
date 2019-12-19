#include <iostream>

#include "ParticlesData.hpp"

namespace sphexa
{
template <typename T>
struct EvrardCollapseInputFileReader
{
    static ParticlesDataEvrard<T> load(const size_t size, const std::string &filename)
    {
        ParticlesDataEvrard<T> pd;

#if defined(USE_MPI)
        pd.n = size;
        pd.comm = MPI_COMM_WORLD;
        MPI_Comm_size(pd.comm, &pd.nrank);
        MPI_Comm_rank(pd.comm, &pd.rank);
        MPI_Get_processor_name(pd.pname, &pd.pnamelen);

        loadDataFromFileMPI(filename, pd);
#else
        loadDataFromFile(size, filename, pd);
#endif
        return pd;
    }

#if defined(USE_MPI)
    static void loadDataFromFileMPI(const std::string &filename, ParticlesDataEvrard<T> &pd)
    {
        const size_t split = pd.n / pd.nrank;
        const size_t remaining = pd.n - pd.nrank * split;

        pd.count = pd.rank == 0 ? split : split + remaining;
        pd.resize(pd.count);

        MPI_File fh;
        MPI_Status status;

        const MPI_Offset col = pd.n * sizeof(double);

        MPI_Offset offset = pd.rank * split * sizeof(double);
        if (pd.rank > 0) offset += remaining * sizeof(double);

        if (pd.rank == 0) printf("Loading input file for Evrard Collapse... ");

        int err = MPI_File_open(pd.comm, filename.c_str(), MPI_MODE_RDWR, MPI_INFO_NULL, &fh);
        if (err != MPI_SUCCESS)
        {
            if (pd.rank == 0) printf("Error %d! Can't open the file!\n", err);
            MPI_Abort(pd.comm, err);
            exit(EXIT_FAILURE);
        }


        MPI_File_set_view(fh, offset, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
        MPI_File_read(fh, &pd.x[0], pd.count, MPI_DOUBLE, &status);

        MPI_File_set_view(fh, 1 * col + offset, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
        MPI_File_read(fh, &pd.y[0], pd.count, MPI_DOUBLE, &status);

        MPI_File_set_view(fh, 2 * col + offset, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
        MPI_File_read(fh, &pd.z[0], pd.count, MPI_DOUBLE, &status);

        MPI_File_set_view(fh, 3 * col + offset, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
        MPI_File_read(fh, &pd.vx[0], pd.count, MPI_DOUBLE, &status);

        MPI_File_set_view(fh, 4 * col + offset, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
        MPI_File_read(fh, &pd.vy[0], pd.count, MPI_DOUBLE, &status);

        MPI_File_set_view(fh, 5 * col + offset, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
        MPI_File_read(fh, &pd.vz[0], pd.count, MPI_DOUBLE, &status);

        MPI_File_set_view(fh, 6 * col + offset, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
        MPI_File_read(fh, &pd.ro[0], pd.count, MPI_DOUBLE, &status);

        MPI_File_set_view(fh, 7 * col + offset, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
        MPI_File_read(fh, &pd.u[0], pd.count, MPI_DOUBLE, &status);

        MPI_File_set_view(fh, 8 * col + offset, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
        MPI_File_read(fh, &pd.p[0], pd.count, MPI_DOUBLE, &status);

        MPI_File_set_view(fh, 9 * col + offset, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
        MPI_File_read(fh, &pd.h[0], pd.count, MPI_DOUBLE, &status);

        MPI_File_set_view(fh, 10 * col + offset, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
        MPI_File_read(fh, &pd.m[0], pd.count, MPI_DOUBLE, &status);

        MPI_File_close(&fh);

        if (pd.rank == 0) printf("OK\n");
        init(pd);
    }
#else
    static ParticlesDataEvrard<T> loadDataFromFile(const size_t size, const std::string &filename, ParticlesDataEvrard<T> &pd)
    {
        pd.resize(size);

        pd.n = pd.x.size();
        pd.count = pd.x.size();

        printf("Loading input file for Evrard Collapse... ");

        std::ifstream inputfile(filename, std::ios::binary);

        if (inputfile.is_open())
        {
            inputfile.read(reinterpret_cast<char *>(pd.x.data()), sizeof(T) * pd.x.size());
            inputfile.read(reinterpret_cast<char *>(pd.y.data()), sizeof(T) * pd.y.size());
            inputfile.read(reinterpret_cast<char *>(pd.z.data()), sizeof(T) * pd.z.size());
            inputfile.read(reinterpret_cast<char *>(pd.vx.data()), sizeof(T) * pd.vx.size());
            inputfile.read(reinterpret_cast<char *>(pd.vy.data()), sizeof(T) * pd.vy.size());
            inputfile.read(reinterpret_cast<char *>(pd.vz.data()), sizeof(T) * pd.vz.size());
            inputfile.read(reinterpret_cast<char *>(pd.ro.data()), sizeof(T) * pd.ro.size());
            inputfile.read(reinterpret_cast<char *>(pd.u.data()), sizeof(T) * pd.u.size());
            inputfile.read(reinterpret_cast<char *>(pd.p.data()), sizeof(T) * pd.p.size());
            inputfile.read(reinterpret_cast<char *>(pd.h.data()), sizeof(T) * pd.h.size());
            inputfile.read(reinterpret_cast<char *>(pd.m.data()), sizeof(T) * pd.m.size());

            inputfile.close();

            init(pd);
        }
        else
            printf("ERROR: Can't open file %s\n", filename);

            printf("OK\n");

        return pd;
    }
#endif

    static void init(ParticlesDataEvrard<T> &pd)
    {
        // additional fields for evrard
        std::fill(pd.temp.begin(), pd.temp.end(), 1.0);
        std::fill(pd.mue.begin(), pd.mue.end(), 2.0);
        std::fill(pd.mui.begin(), pd.mui.end(), 10.0);
        std::fill(pd.vx.begin(), pd.vx.end(), 0.0);
        std::fill(pd.vy.begin(), pd.vy.end(), 0.0);
        std::fill(pd.vz.begin(), pd.vz.end(), 0.0);
        std::fill(pd.grad_P_x.begin(), pd.grad_P_x.end(), 0.0);
        std::fill(pd.grad_P_y.begin(), pd.grad_P_y.end(), 0.0);
        std::fill(pd.grad_P_z.begin(), pd.grad_P_z.end(), 0.0);
        std::fill(pd.du.begin(), pd.du.end(), 0.0);
        std::fill(pd.du_m1.begin(), pd.du_m1.end(), 0.0);
        std::fill(pd.dt.begin(), pd.dt.end(), 0.0001);
        std::fill(pd.dt_m1.begin(), pd.dt_m1.end(), 0.0001);

        std::fill(pd.fx.begin(), pd.fx.end(), 0);
        std::fill(pd.fy.begin(), pd.fy.end(), 0);
        std::fill(pd.fz.begin(), pd.fz.end(), 0);
        std::fill(pd.ugrav.begin(), pd.ugrav.end(), 0);

        for (unsigned int i = 0; i < pd.count; i++)
        {
            pd.x_m1[i] = pd.x[i] - pd.vx[i] * pd.dt[0];
            pd.y_m1[i] = pd.y[i] - pd.vy[i] * pd.dt[0];
            pd.z_m1[i] = pd.z[i] - pd.vz[i] * pd.dt[0];
        }
        pd.etot = pd.ecin = pd.eint = pd.egrav = 0.0;
        pd.minDt = 1e-4;
    }
};
} // namespace sphexa
