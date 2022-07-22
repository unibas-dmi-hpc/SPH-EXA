#pragma once

#include <map>
#include <string>
#include <vector>

#include "cstone/sfc/box.hpp"

#include "file_utils.hpp"
#include "mpi_file_utils.hpp"
#include "sph/particles_data.hpp"

namespace sphexa
{

template<typename Dataset>
struct IFileWriter
{
    virtual void dump(Dataset& d, size_t firstIndex, size_t lastIndex,
                      const cstone::Box<typename Dataset::RealType>& box, std::string path) const = 0;
    virtual void constants(const std::map<std::string, double>& c, std::string path) const        = 0;

    virtual ~IFileWriter() = default;
};

template<class Dataset>
struct AsciiWriter : public IFileWriter<Dataset>
{
    void dump(Dataset& d, size_t firstIndex, size_t lastIndex, const cstone::Box<typename Dataset::RealType>& box,
              std::string path) const override
    {
        const char separator = ' ';
        path += std::to_string(d.ttot) + ".txt";

        int rank, numRanks;
        MPI_Comm_rank(d.comm, &rank);
        MPI_Comm_size(d.comm, &numRanks);

        for (int turn = 0; turn < numRanks; turn++)
        {
            if (turn == rank)
            {
                try
                {
                    auto fieldPointers = getOutputArrays(d);

                    bool append = rank != 0;
                    fileutils::writeAscii(firstIndex, lastIndex, path, append, fieldPointers, separator);
                }
                catch (std::runtime_error& ex)
                {
                    if (rank == 0) fprintf(stderr, "ERROR: %s Terminating\n", ex.what());
                    MPI_Abort(d.comm, 1);
                }
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    void constants(const std::map<std::string, double>& c, std::string path) const override {}
};

template<class Dataset>
struct H5PartWriter : public IFileWriter<Dataset>
{
    void dump(Dataset& d, size_t firstIndex, size_t lastIndex, const cstone::Box<typename Dataset::RealType>& box,
              std::string path) const override
    {
#ifdef SPH_EXA_HAVE_H5PART
        path += ".h5part";
        fileutils::writeH5Part(d, firstIndex, lastIndex, box, path);
#else
        throw std::runtime_error("Cannot write to HDF5 file: H5Part not enabled\n");
#endif
    }

    /*! @brief write simulation parameters to file
     *
     * @param c        (name, value) pairs of constants
     * @param path     path to HDF5 file
     *
     * Note: file is being opened serially, must be called on one rank only.
     */
    void constants(const std::map<std::string, double>& c, std::string path) const override
    {
#ifdef SPH_EXA_HAVE_H5PART
        path += ".h5part";
        const char* h5_fname = path.c_str();

        if (std::filesystem::exists(h5_fname))
        {
            throw std::runtime_error("Cannot write constants: file " + path + " already exists\n");
        }

        H5PartFile* h5_file = nullptr;
        h5_file             = H5PartOpenFile(h5_fname, H5PART_WRITE);

        for (auto it = c.begin(); it != c.end(); ++it)
        {
            H5PartWriteFileAttrib(h5_file, it->first.c_str(), H5PART_FLOAT64, &(it->second), 1);
        }

        H5PartCloseFile(h5_file);
#else
        throw std::runtime_error("Cannot write to HDF5 file: H5Part not enabled\n");
#endif
    }
};

template<class Dataset>
std::unique_ptr<IFileWriter<Dataset>> fileWriterFactory(bool ascii)
{
    if (ascii) { return std::make_unique<AsciiWriter<Dataset>>(); }
    else { return std::make_unique<H5PartWriter<Dataset>>(); }
}

} // namespace sphexa
