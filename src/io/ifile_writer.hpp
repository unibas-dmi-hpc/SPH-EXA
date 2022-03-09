#pragma once

#include <string>
#include <vector>

#include "file_utils.hpp"
#include "mpi_file_utils.hpp"
#include "sph/particles_data.hpp"

namespace sphexa
{

template<typename Dataset>
struct IFileWriter
{
    virtual void dump(Dataset& d, size_t firstIndex, size_t lastIndex, std::string path) const = 0;
    virtual void constants(const std::vector<std::string>& names, const std::vector<double>& values,
                           std::string path) const                                             = 0;

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
                catch (std::runtime_error& ex)
                {
                    if (d.rank == 0) fprintf(stderr, "ERROR: %s. Terminating\n", ex.what());
                    MPI_Abort(d.comm, 1);
                }
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    void constants(const std::vector<std::string>& names, const std::vector<double>& values,
                   std::string path) const override
    {
    }
};

template<class Dataset>
struct H5PartWriter : public IFileWriter<Dataset>
{
    void dump(Dataset& d, size_t firstIndex, size_t lastIndex, std::string path) const override
    {
#ifdef SPH_EXA_HAVE_H5PART
        path += ".h5part";
        fileutils::writeH5Part(d, firstIndex, lastIndex, path);
#else
        throw std::runtime_error("Cannot write to HDF5 file: H5Part not enabled\n");
#endif
    }

    /*! @brief write simulation parameters to file
     *
     * @param names    names of the constants
     * @param values   their values
     * @param path     path to HDF5 file
     *
     * Note: file is being opened serially, must be called on one rank only.
     */
    void constants(const std::vector<std::string>& names, const std::vector<double>& values,
                   std::string path) const override
    {
#ifdef SPH_EXA_HAVE_H5PART
        if (names.size() != values.size())
        {
            throw std::runtime_error("Cannot write constants: name/value size mismatch\n");
        }

        path += ".h5part";
        const char* h5_fname = path.c_str();

        if (std::filesystem::exists(h5_fname))
        {
            throw std::runtime_error("Cannot write constants: file " + path + " already exists\n");
        }

        H5PartFile* h5_file = nullptr;
        h5_file             = H5PartOpenFile(h5_fname, H5PART_WRITE);

        for (size_t i = 0; i < names.size(); ++i)
        {
            H5PartWriteFileAttrib(h5_file, names[i].c_str(), H5PART_FLOAT64, &values[i], 1);
        }

        H5PartCloseFile(h5_file);
#else
        throw std::runtime_error("Cannot write to HDF5 file: H5Part not enabled\n");
#endif
    }
};

} // namespace sphexa
