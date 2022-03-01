#pragma once

#include <string>
#include <vector>

#include "particles_data.hpp"

namespace sphexa
{

template<typename Dataset>
struct IFileWriter
{
    virtual void dump(Dataset& d, size_t firstIndex, size_t lastIndex, std::string path) const = 0;

    virtual ~IFileWriter() = default;
};

template<class Dataset>
struct AsciiWriter : public IFileWriter<Dataset>
{
    void dump(Dataset& d, size_t firstIndex, size_t lastIndex, std::string path) const override
    {
        const char separator = ' ';
        path += std::to_string(d.iteration) + ".txt";

        for (int turn = 0; turn < d.nrank; turn++)
        {
            if (turn == d.rank)
            {
                try
                {
                    auto fieldPointers = getOutputArrays(d);

                    bool append = d.rank != 0;
                    fileutils::writeAscii(firstIndex, lastIndex, path, append, fieldPointers, separator);
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
    void dump(Dataset& d, size_t firstIndex, size_t lastIndex, std::string path) const override
    {
        path += ".h5part";
#ifdef SPH_EXA_HAVE_H5PART
        fileutils::writeH5Part(d, firstIndex, lastIndex, path);
#else
        throw std::runtime_error("Cannot write to HDF5 file: H5Part not enabled\n");
#endif
    }
};

} // namespace sphexa
