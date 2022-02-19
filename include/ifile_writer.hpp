#pragma once

#include <string>
#include <vector>

namespace sphexa
{

template<typename Dataset>
struct IFileWriter
{
    virtual void dump(const Dataset& d, size_t firstIndex, size_t lastIndex, std::string path) const = 0;

    virtual ~IFileWriter() = default;
};

template<class Dataset>
struct AsciiWriter : public IFileWriter<Dataset>
{
    void dump(const Dataset& d, size_t firstIndex, size_t lastIndex, std::string path) const override
    {
        const char separator = ' ';
        path += std::to_string(d.iteration) + ".txt";

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
                catch (MPIFileNotOpenedException& ex)
                {
                    if (d.rank == 0) fprintf(stderr, "ERROR: %s. Terminating\n", ex.what());
                    MPI_Abort(d.comm, ex.mpierr);
                }
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
};

template<class Dataset>
struct H5PartWriter : public IFileWriter<Dataset>
{
    void dump(const Dataset& d, size_t firstIndex, size_t lastIndex, std::string path) const override
    {
        path += ".h5part";
#ifdef SPH_EXA_HAVE_H5PART
        fileutils::writeParticleDataToBinFileWithH5Part(d, firstIndex, lastIndex, path);
#else
        throw std::runtime_error("Cannot write to HDF5 file: H5Part not enabled\n");
#endif
    }
};

} // namespace sphexa
