#pragma once

#include <cmath>

#include <iomanip>

#include "IFileWriter.hpp"


struct NohSolutionWriter
{
    static void dump1DToAsciiFile(const size_t n,
                                  const std::vector<double>& r,
                                  const std::vector<double>& rho,
                                  const std::vector<double>& u,
                                  const std::vector<double>& p,
                                  const std::vector<double>& vel,
                                  const std::string &outfile)
    {
        try
        {
            std::ofstream out(outfile);

            out << " " << std::setw(15) << "r"              // Column 01 : position 1D
                << " " << std::setw(15) << "rho"            // Column 02 : density         (Real value)
                << " " << std::setw(15) << "u"              // Column 03 : internal energy (Real value)
                << " " << std::setw(15) << "p"              // Column 04 : pressure        (Real value)
                << " " << std::setw(15) << "vel"            // Column 05 : velocity 1D     (Real value)
                << std::endl;

            for(size_t i = 0; i < n; i++)
            {
                out << " " << std::setw(15) << std::setprecision(6) << std::scientific << std::uppercase << r[i]   << std::nouppercase
                    << " " << std::setw(15) << std::setprecision(6) << std::scientific << std::uppercase << rho[i] << std::nouppercase
                    << " " << std::setw(15) << std::setprecision(6) << std::scientific << std::uppercase << u[i]   << std::nouppercase
                    << " " << std::setw(15) << std::setprecision(6) << std::scientific << std::uppercase << p[i]   << std::nouppercase
                    << " " << std::setw(15) << std::setprecision(6) << std::scientific << std::uppercase << vel[i] << std::nouppercase
                    << std::endl;
            }

            out.close();
        }
        catch (std::exception &ex)
        {
            fprintf(stderr, "ERROR: %s. Terminating\n", ex.what());
            exit(EXIT_FAILURE);
        }
    }
};

namespace sphexa
{
template <typename Dataset>
struct NohFileWriter : IFileWriter<Dataset>
{
    void dumpParticleDataToBinFile(const Dataset &d, const std::string &path) const override
    {
        try
        {
            printf("Dumping particles data to file at path: %s\n", path.c_str());
            fileutils::writeParticleDataToBinFile(path, d.x, d.y, d.z, d.vx, d.vy, d.vz, d.h, d.ro, d.u, d.p, d.c, d.grad_P_x, d.grad_P_y,
                                                  d.grad_P_z /*, d.radius*/);
        }
        catch (FileNotOpenedException &ex)
        {
            fprintf(stderr, "ERROR: %s. Terminating\n", ex.what());
            exit(EXIT_FAILURE);
        }
    }

    void dumpParticleDataToAsciiFile(const Dataset &d, int firstIndex, int lastIndex, const std::string &path) const override
    {
        try
        {
            const char separator = ' ';

            printf("Dumping particles data to ASCII file at path: %s\n", path.c_str());
            fileutils::writeParticleDataToAsciiFile(firstIndex,
                                                    lastIndex,
                                                    path,
                                                    separator,
                                                    d.x,
                                                    d.y,
                                                    d.z,
                                                    d.vx,
                                                    d.vy,
                                                    d.vz,
                                                    d.h,
                                                    d.ro,
                                                    d.u,
                                                    d.p,
                                                    d.c,
                                                    d.grad_P_x,
                                                    d.grad_P_y,
                                                    d.grad_P_z);
        }
        catch (FileNotOpenedException& ex)
        {
            fprintf(stderr, "ERROR: %s. Terminating\n", ex.what());
            exit(EXIT_FAILURE);
        }
    }

    #ifdef SPH_EXA_HAVE_H5PART
    void dumpParticleDataToH5File(const Dataset &d, const std::vector<int> &clist, const std::string &path) const override
    {
        try
        {
            fileutils::writeParticleDataToBinFileWithH5Part(d, clist, path, d.x, d.y, d.z, d.h, d.ro);
                //d.vx, d.vy, d.vz, d.h, d.ro, d.u, d.p, d.c, d.grad_P_x,
                //d.grad_P_y, d.grad_P_z /*, d.radius*/);
        }
        catch (MPIFileNotOpenedException &ex)
        {
            if (d.rank == 0) fprintf(stderr, "ERROR: %s. Terminating\n", ex.what());
            MPI_Abort(d.comm, ex.mpierr);
        }
    }
    #endif

    void dumpCheckpointDataToBinFile(const Dataset &, const std::string &) const override
    {
        fprintf(stderr, "Warning: dumping checkpoint is not implemented in NohFileWriter, exiting...\n");
        exit(EXIT_FAILURE);
    }
};

#ifdef USE_MPI

template <typename Dataset>
struct NohMPIFileWriter : IFileWriter<Dataset>
{
    #ifdef SPH_EXA_HAVE_H5PART
    void dumpParticleDataToH5File(const Dataset& d, int firstIndex, int lastIndex, const std::string &path) const override
    {
        fileutils::writeParticleDataToBinFileWithH5Part(d, firstIndex, lastIndex, path);
    }
    #endif

    void dumpParticleDataToBinFile(const Dataset &d, const std::string &path) const override
    {
        try
        {
            fileutils::writeParticleDataToBinFileWithMPI(d, path, d.x, d.y, d.z, d.vx, d.vy, d.vz, d.h, d.ro, d.u, d.p, d.c, d.grad_P_x,
                                                         d.grad_P_y, d.grad_P_z);
        }
        catch (MPIFileNotOpenedException &ex)
        {
            if (d.rank == 0) fprintf(stderr, "ERROR: %s. Terminating\n", ex.what());
            MPI_Abort(d.comm, ex.mpierr);
        }
    };

    void dumpParticleDataToAsciiFile(const Dataset& d, int firstIndex, int lastIndex,
                                     const std::string& path) const override
    {
        const char separator = ' ';

        for (int turn = 0; turn < d.nrank; turn++)
        {
            if (turn == d.rank)
            {
                try
                {
                    fileutils::writeParticleDataToAsciiFile(firstIndex,
                                                            lastIndex,
                                                            path,
                                                            d.rank != 0,
                                                            separator,
                                                            d.x,
                                                            d.y,
                                                            d.z,
                                                            d.vx,
                                                            d.vy,
                                                            d.vz,
                                                            d.h,
                                                            d.ro,
                                                            d.u,
                                                            d.p,
                                                            d.c,
                                                            d.grad_P_x,
                                                            d.grad_P_y,
                                                            d.grad_P_z);
                }
                catch (MPIFileNotOpenedException &ex)
                {
                    if (d.rank == 0) fprintf(stderr, "ERROR: %s. Terminating\n", ex.what());
                    MPI_Abort(d.comm, ex.mpierr);
                }

                MPI_Barrier(MPI_COMM_WORLD);
            }
            else
            {
                MPI_Barrier(MPI_COMM_WORLD);
            }
        }
    }

    void dumpCheckpointDataToBinFile(const Dataset &d, const std::string &) const override
    {
        if (d.rank == 0) fprintf(stderr, "Warning: dumping checkpoint is not implemented in NohMPIFileWriter, exiting...\n");
        MPI_Abort(d.comm, MPI_ERR_OTHER);
    }
};

#endif
} // namespace sphexa
