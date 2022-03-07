#pragma once

#include <mpi.h>

#include "exceptions.hpp"

#ifdef SPH_EXA_HAVE_H5PART
#include <filesystem>
#include "H5Part.h"
#endif

namespace sphexa
{
namespace fileutils
{

namespace details
{
void readFileMPI(const MPI_File&, const size_t, const MPI_Offset&, const MPI_Offset&, const int) {}

template<typename Arg, typename... Args>
void readFileMPI(const MPI_File& file, const size_t count, const MPI_Offset& offset, const MPI_Offset& col,
                 const int blockNo, Arg& first, Args&&... args)
{
    MPI_Status status;

    MPI_File_set_view(file, blockNo * col + offset, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
    MPI_File_read(file, first.data(), count, MPI_DOUBLE, &status);

    readFileMPI(file, count, offset, col, blockNo + 1, args...);
}
} // namespace details

template<typename Dataset, typename... Args>
void readParticleDataFromBinFileWithMPI(const std::string& path, Dataset& pd, Args&&... data)
{
    const size_t split     = pd.n / pd.nrank;
    const size_t remaining = pd.n - pd.nrank * split;

    pd.count = pd.rank != pd.nrank - 1 ? split : split + remaining;
    resize(pd, pd.count);

    MPI_File fh;

    const MPI_Offset col = pd.n * sizeof(double);

    MPI_Offset offset = pd.rank * split * sizeof(double);
    if (pd.rank > pd.nrank - 1) offset += remaining * sizeof(double);

    const int err = MPI_File_open(pd.comm, path.c_str(), MPI_MODE_RDWR, MPI_INFO_NULL, &fh);
    if (err != MPI_SUCCESS) { throw MPIFileNotOpenedException("Can't open MPI file at path: " + path, err); }

    details::readFileMPI(fh, pd.count, offset, col, 0, data...);

    MPI_File_close(&fh);
}

#ifdef SPH_EXA_HAVE_H5PART

inline void writeH5PartField(H5PartFile* h5_file, const std::string& fieldName, const float* field)
{
    static_assert(std::is_same_v<float, h5part_float32_t>);
    H5PartWriteDataFloat32(h5_file, fieldName.c_str(), field);
}

inline void writeH5PartField(H5PartFile* h5_file, const std::string& fieldName, const double* field)
{
    static_assert(std::is_same_v<double, h5part_float64_t>);
    H5PartWriteDataFloat64(h5_file, fieldName.c_str(), field);
}

template<typename Dataset>
void writeH5Part(Dataset& d, size_t firstIndex, size_t lastIndex, const std::string& path)
{
    using h5_int64_t = h5part_int64_t;
    using h5_id_t    = h5part_int64_t;

    // output name
    const char* h5_fname = path.c_str();
    H5PartFile* h5_file  = nullptr;

#ifdef H5PART_PARALLEL_IO
    if (std::filesystem::exists(h5_fname)) { h5_file = H5PartOpenFileParallel(h5_fname, H5PART_APPEND, d.comm); }
    else
    {
        h5_file = H5PartOpenFileParallel(h5_fname, H5PART_WRITE, d.comm);
    }
#else
    if (d.nrank > 1)
    {
        throw std::runtime_error("Cannot write HDF5 output with multiple ranks without parallel HDF5 support\n");
    }
    if (std::filesystem::exists(h5_fname)) { h5_file = H5PartOpenFile(h5_fname, H5PART_APPEND); }
    else
    {
        h5_file = H5PartOpenFile(h5_fname, H5PART_WRITE);
    }
#endif

    // create the current step
    const h5_id_t h5_step = d.iteration;
    H5PartSetStep(h5_file, h5_step);

    H5PartWriteStepAttrib(h5_file, "time", H5PART_FLOAT64, &d.ttot, 1);
    H5PartWriteStepAttrib(h5_file, "step", H5PART_INT64, &h5_step, 1);

    // set number of particles that each rank will write
    const h5_int64_t h5_num_particles = lastIndex - firstIndex;

    // set number of particles that each rank will write
    H5PartSetNumParticles(h5_file, h5_num_particles);

    auto fieldPointers = getOutputArrays(d);
    for (size_t fidx = 0; fidx < fieldPointers.size(); ++fidx)
    {
        const std::string& fieldName = Dataset::fieldNames[d.outputFields[fidx]];
        writeH5PartField(h5_file, fieldName, fieldPointers[fidx] + firstIndex);
    }

    H5PartCloseFile(h5_file);
}

#endif

} // namespace fileutils
} // namespace sphexa
