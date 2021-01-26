#pragma once

#include "IFileWriter.hpp"

namespace sphexa
{
template <typename Dataset>
struct SedovFileWriter : IFileWriter<Dataset>
{
// {{{ dumpParticleDataToBinFile
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
// }}}

// {{{ dumpParticleDataToAsciiFile
    void dumpParticleDataToAsciiFile(const Dataset &d, const std::vector<int> &clist, const std::string &path) const override
    {
        try
        {
            const char separator = ' ';

            printf("Dumping particles data to ASCII file at path: %s\n", path.c_str());
            fileutils::writeParticleDataToAsciiFile(clist, path, separator, d.x, d.y, d.z, d.vx, d.vy, d.vz, d.h, d.ro, d.u, d.p, d.c,
                                                    d.grad_P_x, d.grad_P_y, d.grad_P_z /*, d.radius*/);
        }
        catch (FileNotOpenedException &ex)
        {
            fprintf(stderr, "ERROR: %s. Terminating\n", ex.what());
            exit(EXIT_FAILURE);
        }
    }
// }}}

// {{{ dumpCheckpointDataToBinFile
    void dumpCheckpointDataToBinFile(const Dataset &, const std::string &) const override
    {
        fprintf(stderr, "Warning: dumping checkpoint is not implemented in SedovFileWriter, exiting...\n");
        exit(EXIT_FAILURE);
    }
// }}}
};

#ifdef USE_MPI

template <typename Dataset>
struct SedovMPIFileWriter : IFileWriter<Dataset>
{
// {{{ dumpParticleDataToBinFile
    void dumpParticleDataToBinFile(const Dataset &d, const std::string &path) const override
    {
        try
        {
            fileutils::writeParticleDataToBinFileWithMPI(d, path, d.x, d.y, d.z, d.vx, d.vy, d.vz, d.h, d.ro, d.u, d.p, d.c, d.grad_P_x,
                                                         d.grad_P_y, d.grad_P_z /*, d.radius*/);
        }
        catch (MPIFileNotOpenedException &ex)
        {
            if (d.rank == 0) fprintf(stderr, "ERROR: %s. Terminating\n", ex.what());
            MPI_Abort(d.comm, ex.mpierr);
        }
    };
// }}}

// {{{ dumpParticleDataToAsciiFile
    void dumpParticleDataToAsciiFile(const Dataset &d, const std::vector<int> &clist, const std::string &path) const override
    {
        const char separator = ' ';

        for (int turn = 0; turn < d.nrank; turn++)
        {
            if (turn == d.rank)
            {
                try
                {
                    fileutils::writeParticleDataToAsciiFile(clist, path, d.rank != 0, separator, d.x, d.y, d.z, d.vx, d.vy, d.vz, d.h, d.ro, d.u, d.p,
                                                            d.c, d.grad_P_x, d.grad_P_y, d.grad_P_z /*, d.radius*/);
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
// }}}

// {{{ dumpParticleDataToH5File
#ifdef USE_H5
    void dumpParticleDataToH5File(const Dataset &d, const std::vector<int> &clist, const std::string &path) const override
    {
        try
        {
            fileutils::writeParticleDataToBinFileWithH5Part(d, clist, path, \
                d.x, d.y, d.z, d.h, d.ro);
//             d.vx, d.vy, d.vz, d.h, d.ro, d.u, d.p, d.c, d.grad_P_x,
//             d.grad_P_y, d.grad_P_z /*, d.radius*/);
        }
        catch (MPIFileNotOpenedException &ex)
        {
            if (d.rank == 0) fprintf(stderr, "ERROR: %s. Terminating\n", ex.what());
            MPI_Abort(d.comm, ex.mpierr);
        }
    };
#endif
// }}}

// {{{ dumpCheckpointDataToBinFile
    void dumpCheckpointDataToBinFile(const Dataset &d, const std::string &) const override
    {
        if (d.rank == 0) fprintf(stderr, "Warning: dumping checkpoint is not implemented in SedovMPIFileWriter, exiting...\n");
        MPI_Abort(d.comm, MPI_ERR_OTHER);
    }
// }}}
};

#endif
} // namespace sphexa
