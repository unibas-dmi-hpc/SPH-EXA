#include <iostream>

#include "particles_data.hpp"
#include "ifile_reader.hpp"

namespace sphexa
{

template<typename Dataset>
struct EvrardFileReader : IFileReader<Dataset>
{
    Dataset read(const std::string& path, const size_t noParticles) const override
    {
        Dataset d;
        d.n = noParticles;
        initMPIData(d);

        try
        {
            fileutils::readParticleDataFromBinFileWithMPI(
                path, d, d.x, d.y, d.z, d.vx, d.vy, d.vz, d.rho, d.u, d.p, d.h, d.m);
            if (d.rank == 0)
                printf("Loaded input file with %lu particles for Evrard Collapse from path '%s' \n", d.n, path.c_str());
        }
        catch (MPIFileNotOpenedException& ex)
        {
            if (d.rank == 0) fprintf(stderr, "ERROR: %s. Terminating\n", ex.what());
            MPI_Abort(d.comm, ex.mpierr);
        }

        this->init(d);

        return d;
    }

private:
    void initMPIData(Dataset& d) const
    {
        d.comm = MPI_COMM_WORLD;
        MPI_Comm_size(d.comm, &d.nrank);
        MPI_Comm_rank(d.comm, &d.rank);
    }

    void init(Dataset& pd) const
    {
        // additional fields for evrard
        std::fill(pd.mue.begin(), pd.mue.end(), 2.0);
        std::fill(pd.mui.begin(), pd.mui.end(), 10.0);
        std::fill(pd.vx.begin(), pd.vx.end(), 0.0);
        std::fill(pd.vy.begin(), pd.vy.end(), 0.0);
        std::fill(pd.vz.begin(), pd.vz.end(), 0.0);
        std::fill(pd.du_m1.begin(), pd.du_m1.end(), 0.0);
        std::fill(pd.dt_m1.begin(), pd.dt_m1.end(), 0.0001);

        for (size_t i = 0; i < pd.count; ++i)
        {
            pd.x_m1[i] = pd.x[i] - pd.vx[i] * pd.dt[0];
            pd.y_m1[i] = pd.y[i] - pd.vy[i] * pd.dt[0];
            pd.z_m1[i] = pd.z[i] - pd.vz[i] * pd.dt[0];
        }
        pd.etot = pd.ecin = pd.eint = pd.egrav = 0.0;
        pd.minDt                               = 1e-4;
    }
};

} // namespace sphexa
