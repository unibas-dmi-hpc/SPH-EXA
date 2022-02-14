#include <iostream>

#include "ifile_reader.hpp"
#include "file_utils.hpp"

namespace sphexa
{

template<typename Dataset>
struct SedovDataFileReader : IFileReader<Dataset>
{
    Dataset readParticleDataFromBinFile(const std::string& path, const size_t noParticles) const override
    {
        Dataset d;
        d.n = noParticles;

        d.resize(noParticles);
        d.count = d.x.size();

        try
        {
            init(d);

            printf("Loading input file with %lu particles at path '%s'... ", d.n, path.c_str());
            fileutils::readParticleDataFromBinFile(path,
                                                   d.x,
                                                   d.y,
                                                   d.z,
                                                   d.vx,
                                                   d.vy,
                                                   d.vz,
                                                   d.ro,
                                                   d.u,
                                                   d.p,
                                                   d.h,
                                                   d.m,
                                                   d.temp,
                                                   d.mue,
                                                   d.mui,
                                                   d.du,
                                                   d.du_m1,
                                                   d.dt,
                                                   d.dt_m1,
                                                   d.x_m1,
                                                   d.y_m1,
                                                   d.z_m1,
                                                   d.c,
                                                   d.grad_P_x,
                                                   d.grad_P_y,
                                                   d.grad_P_z);
            printf("OK\n");
        }
        catch (FileNotOpenedException& ex)
        {
            printf("ERROR: %s. Terminating\n", ex.what());
            exit(EXIT_FAILURE);
        }

        return d;
    }

    Dataset readParticleDataFromCheckpointBinFile(const std::string& path) const override
    {
        Dataset       d;
        std::ifstream inputfile(path, std::ios::binary);

        if (inputfile.is_open())
        {
            inputfile.read(reinterpret_cast<char*>(&d.n), sizeof(size_t));

            d.resize(d.n);

            d.n     = d.x.size();
            d.count = d.x.size();

            printf("Loading checkpoint file with %lu particles ... ", d.n);

            inputfile.read(reinterpret_cast<char*>(&d.ttot), sizeof(d.ttot));
            inputfile.read(reinterpret_cast<char*>(&d.minDt), sizeof(d.minDt));

            fileutils::details::readParticleDataFromBinFile(inputfile,
                                                            d.x,
                                                            d.y,
                                                            d.z,
                                                            d.vx,
                                                            d.vy,
                                                            d.vz,
                                                            d.ro,
                                                            d.u,
                                                            d.p,
                                                            d.h,
                                                            d.m,
                                                            d.temp,
                                                            d.mue,
                                                            d.mui,
                                                            d.du,
                                                            d.du_m1,
                                                            d.dt,
                                                            d.dt_m1,
                                                            d.x_m1,
                                                            d.y_m1,
                                                            d.z_m1,
                                                            d.c,
                                                            d.grad_P_x,
                                                            d.grad_P_y,
                                                            d.grad_P_z);

            d.etot = d.ecin = d.eint = d.egrav = 0.0;

            inputfile.close();

            printf("OK\n");
        }
        else
            printf("ERROR: Can't open file %s\n", path.c_str());
        return d;
    }

protected:
    void init(Dataset& d) const
    {
        std::fill(d.temp.begin(), d.temp.end(), 1.0);

        std::fill(d.mue.begin(), d.mue.end(), 2.0);
        std::fill(d.mui.begin(), d.mui.end(), 10.0);

        std::fill(d.vx.begin(), d.vx.end(), 0.0);
        std::fill(d.vy.begin(), d.vy.end(), 0.0);
        std::fill(d.vz.begin(), d.vz.end(), 0.0);

        std::fill(d.grad_P_x.begin(), d.grad_P_x.end(), 0.0);
        std::fill(d.grad_P_y.begin(), d.grad_P_y.end(), 0.0);
        std::fill(d.grad_P_z.begin(), d.grad_P_z.end(), 0.0);

        std::fill(d.du.begin(), d.du.end(), 0.0);
        std::fill(d.du_m1.begin(), d.du_m1.end(), 0.0);

        std::fill(d.dt.begin(), d.dt.end(), 0.0001);
        std::fill(d.dt_m1.begin(), d.dt_m1.end(), 0.0001);

        for (size_t i = 0; i < d.count; ++i)
        {
            d.x_m1[i] = d.x[i] - d.vx[i] * d.dt[0];
            d.y_m1[i] = d.y[i] - d.vy[i] * d.dt[0];
            d.z_m1[i] = d.z[i] - d.vz[i] * d.dt[0];
        }

        d.etot  = 0.0;
        d.ecin  = 0.0;
        d.eint  = 0.0;
        d.egrav = 0.0;

        d.minDt = 1e-4;
    }
};

#ifdef USE_MPI
template<typename Dataset>
struct SedovDataMPIFileReader : SedovDataFileReader<Dataset>
{
    Dataset readParticleDataFromBinFile(const std::string& path, const size_t noParticles) const override
    {
        Dataset d;
        d.n = noParticles;

        initMPIData(d);

        try
        {
            this->init(d);

            fileutils::readParticleDataFromBinFileWithMPI(path,
                                                          d,
                                                          d.x,
                                                          d.y,
                                                          d.z,
                                                          d.vx,
                                                          d.vy,
                                                          d.vz,
                                                          d.ro,
                                                          d.u,
                                                          d.p,
                                                          d.h,
                                                          d.m,
                                                          d.temp,
                                                          d.mue,
                                                          d.mui,
                                                          d.du,
                                                          d.du_m1,
                                                          d.dt,
                                                          d.dt_m1,
                                                          d.x_m1,
                                                          d.y_m1,
                                                          d.z_m1,
                                                          d.c,
                                                          d.grad_P_x,
                                                          d.grad_P_y,
                                                          d.grad_P_z);

            if (d.rank == 0) printf("Loaded input file with %lu particles from path '%s' \n", d.n, path.c_str());
        }
        catch (MPIFileNotOpenedException& ex)
        {
            if (d.rank == 0) fprintf(stderr, "ERROR: %s. Terminating\n", ex.what());
            MPI_Abort(d.comm, ex.mpierr);
        }

        return d;
    }

    Dataset readParticleDataFromCheckpointBinFile(const std::string& path) const override
    {
        Dataset d;

        initMPIData(d);

        try
        {
            fileutils::readParticleCheckpointDataFromBinFileWithMPI(path,
                                                                    d,
                                                                    d.x,
                                                                    d.y,
                                                                    d.z,
                                                                    d.vx,
                                                                    d.vy,
                                                                    d.vz,
                                                                    d.ro,
                                                                    d.u,
                                                                    d.p,
                                                                    d.h,
                                                                    d.m,
                                                                    d.temp,
                                                                    d.mue,
                                                                    d.mui,
                                                                    d.du,
                                                                    d.du_m1,
                                                                    d.dt,
                                                                    d.dt_m1,
                                                                    d.x_m1,
                                                                    d.y_m1,
                                                                    d.z_m1,
                                                                    d.c,
                                                                    d.grad_P_x,
                                                                    d.grad_P_y,
                                                                    d.grad_P_z);

            d.etot = d.ecin = d.eint = d.egrav = 0.0;

            if (d.rank == 0) printf("Loaded checkpoint file with %lu particles from path '%s'\n", d.n, path.c_str());
        }
        catch (MPIFileNotOpenedException& ex)
        {
            if (d.rank == 0) fprintf(stderr, "ERROR: %s. Terminating\n", ex.what());
            MPI_Abort(d.comm, ex.mpierr);
        }

        return d;
    }

private:
    void initMPIData(Dataset& d) const
    {
        d.comm = MPI_COMM_WORLD;
        MPI_Comm_size(d.comm, &d.nrank);
        MPI_Comm_rank(d.comm, &d.rank);
        MPI_Get_processor_name(d.pname, &d.pnamelen);
    }
};
#endif

} // namespace sphexa
