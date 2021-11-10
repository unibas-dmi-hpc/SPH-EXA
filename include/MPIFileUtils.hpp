#pragma once

#include <mpi.h>

#include "Exceptions.hpp"

#ifdef SPH_EXA_HAVE_H5PART
# include <filesystem>
# include "H5Part.h"
#endif

namespace sphexa
{
namespace fileutils
{
// {{{ namespace details
namespace details
{
void readFileMPI(const MPI_File &, const size_t, const MPI_Offset &, const MPI_Offset &, const int) {}

template <typename Arg, typename... Args>
void readFileMPI(const MPI_File &file, const size_t count, const MPI_Offset &offset, const MPI_Offset &col, const int blockNo, Arg &first,
                 Args &&... args)
{
    MPI_Status status;

    MPI_File_set_view(file, blockNo * col + offset, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
    MPI_File_read(file, first.data(), count, MPI_DOUBLE, &status);

    readFileMPI(file, count, offset, col, blockNo + 1, args...);
}

void writeParticleDataToBinFileWithMPI(const MPI_File &, const size_t, const MPI_Offset &, const MPI_Offset &, const int) {}

template <typename Arg, typename... Args>
void writeParticleDataToBinFileWithMPI(const MPI_File &file, const size_t count, const MPI_Offset &offset, const MPI_Offset &col,
                                       const int blockNo, Arg &first, Args &&... args)
{
    MPI_Status status;
    MPI_File_set_view(file, blockNo * col + offset, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
    MPI_File_write(file, first.data(), count, MPI_DOUBLE, &status);
    writeParticleDataToBinFileWithMPI(file, count, offset, col, blockNo + 1, args...);
}

#ifdef SPH_EXA_HAVE_H5PART
template <typename Dataset, typename... Args>
void writeParticleDataToBinFileWithH5Part(const Dataset &, const std::vector<int> &, const std::string &, Args &&... ){}
#endif
} // namespace details
// }}}

// {{{ writeParticleCheckpointDataToBinWithMPI
template <typename Dataset, typename... Args>
void writeParticleCheckpointDataToBinWithMPI(const Dataset &d, const std::string &path, Args &&... data)
{
    MPI_File file;
    MPI_Status status;

    const int err = MPI_File_open(d.comm, path.c_str(), MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &file);
    if (err != MPI_SUCCESS) { throw MPIFileNotOpenedException("Can't open MPI file at path: " + path, err); }

    const size_t split = d.n / d.nrank;
    const size_t remaining = d.n - d.nrank * split;

    const MPI_Offset col = d.n * sizeof(double);
    const MPI_Offset headerOffset = 2 * sizeof(double) + sizeof(size_t);
    MPI_Offset offset = headerOffset + d.rank * split * sizeof(double);

    if (d.rank > d.nrank - 1) offset += remaining * sizeof(double);
    if (d.rank == 0)
    {
        MPI_File_write(file, &d.n, 1, MPI_UNSIGNED_LONG, &status);
        MPI_File_write(file, &d.ttot, 1, MPI_DOUBLE, &status);
        MPI_File_write(file, &d.minDt, 1, MPI_DOUBLE, &status);
    }
    MPI_Barrier(d.comm);

    details::writeParticleDataToBinFileWithMPI(file, d.count, offset, col, 0, data...);

    MPI_File_close(&file);
}
// }}}

// {{{ writeParticleDataToBinFileWithMPI
template <typename Dataset, typename... Args>
void writeParticleDataToBinFileWithMPI(const Dataset &d, const std::string &path, Args &&... data)
{
    MPI_File file;

    const int err = MPI_File_open(d.comm, path.c_str(), MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &file);
    if (err != MPI_SUCCESS) { throw MPIFileNotOpenedException("Can't open MPI file at path: " + path, err); }

    const size_t split = d.n / d.nrank;
    const size_t remaining = d.n - d.nrank * split;

    const MPI_Offset col = d.n * sizeof(double);
    MPI_Offset offset = d.rank * split * sizeof(double);
    if (d.rank > d.nrank - 1) offset += remaining * sizeof(double);

    details::writeParticleDataToBinFileWithMPI(file, d.count, offset, col, 0, data...);

    MPI_File_close(&file);
}
// }}}

// {{{ writeParticleDataToBinFileWithH5Part
#ifdef SPH_EXA_HAVE_H5PART
template<typename Dataset, typename... Args>
void writeParticleDataToBinFileWithH5Part(const Dataset& d, int firstIndex, int lastIndex, const std::string& path,
                                          Args&&... data)
{
    using h5_int64_t = h5part_int64_t;
    using h5_id_t    = h5part_int64_t;

    // verbosity level: H5_VERBOSE_DEBUG/H5_VERBOSE_INFO/H5_VERBOSE_DEFAULT
//    const h5_int64_t h5_verbosity = H5_VERBOSE_DEFAULT;
//    H5AbortOnError();
    // output name
    const char* h5_fname = path.c_str();
    H5PartFile *h5_file = nullptr;
    // open file
    if (std::filesystem::exists(h5_fname)) {
        h5_file = H5PartOpenFile(h5_fname, H5PART_APPEND);
    }
    else {
        h5_file = H5PartOpenFile(h5_fname, H5PART_WRITE);
    }

    // create first step
    const h5_id_t h5_step = d.iteration;
    H5PartSetStep(h5_file, h5_step);

    // get number of particles that each rank will write
    const int h5_begin = firstIndex;
    const int h5_end   = lastIndex;
    const h5_int64_t h5_num_particles = h5_end - h5_begin + 1;

    // set number of particles that each rank will write
    H5PartSetNumParticles(h5_file, h5_num_particles);

    h5part_float64_t h5_data_x[h5_num_particles];
    h5part_float64_t h5_data_y[h5_num_particles];
    h5part_float64_t h5_data_z[h5_num_particles];
    h5part_float64_t h5_data_h[h5_num_particles];
    h5part_float64_t h5_data_ro[h5_num_particles];
    // TODO:
    //   vector<T>::const_iterator first = myVec.begin() + 100000;
    //   vector<T>::const_iterator last = myVec.begin() + 101000;
    //   vector<T> newVec(first, last);
#pragma omp parallel for
    for (auto ii = h5_begin; ii < h5_end; ii++) {
        auto jj = ii - h5_begin;
        h5_data_x[jj] = d.x[ii];
        h5_data_y[jj] = d.y[ii];
        h5_data_z[jj] = d.z[ii];
        h5_data_h[jj] = d.h[ii];
        h5_data_ro[jj] = d.ro[ii];
    }
    // write data
    // MPI_Barrier(MPI_COMM_WORLD);
    H5PartWriteDataFloat64(h5_file, "x", h5_data_x);
    H5PartWriteDataFloat64(h5_file, "y", h5_data_y);
    H5PartWriteDataFloat64(h5_file, "z", h5_data_z);
    H5PartWriteDataFloat64(h5_file, "h", h5_data_h);
    H5PartWriteDataFloat64(h5_file, "ro", h5_data_ro);
    H5PartCloseFile(h5_file);
}
#endif
// }}}

// {{{ readParticleDataFromBinFileWithMPI
template <typename Dataset, typename... Args>
void readParticleDataFromBinFileWithMPI(const std::string &path, Dataset &pd, Args &&... data)
{
    const size_t split = pd.n / pd.nrank;
    const size_t remaining = pd.n - pd.nrank * split;

    pd.count = pd.rank != pd.nrank - 1 ? split : split + remaining;
    pd.resize(pd.count);

    MPI_File fh;

    const MPI_Offset col = pd.n * sizeof(double);

    MPI_Offset offset = pd.rank * split * sizeof(double);
    if (pd.rank > pd.nrank - 1) offset += remaining * sizeof(double);

    const int err = MPI_File_open(pd.comm, path.c_str(), MPI_MODE_RDWR, MPI_INFO_NULL, &fh);
    if (err != MPI_SUCCESS) { throw MPIFileNotOpenedException("Can't open MPI file at path: " + path, err); }

    details::readFileMPI(fh, pd.count, offset, col, 0, data...);

    MPI_File_close(&fh);
}
// }}}

// {{{ readParticleCheckpointDataFromBinFileWithMPI
template <typename Dataset, typename... Args>
void readParticleCheckpointDataFromBinFileWithMPI(const std::string &path, Dataset &pd, Args &&... data)
{
    MPI_File fh;
    MPI_Status status;

    const int err = MPI_File_open(pd.comm, path.c_str(), MPI_MODE_RDWR, MPI_INFO_NULL, &fh);
    if (err != MPI_SUCCESS)
    {
        throw MPIFileNotOpenedException("Can't open MPI file at path: " + path, err);
    }

    MPI_File_read(fh, &pd.n, 1, MPI_UNSIGNED_LONG, &status);

    const MPI_Offset headerOffset = 2 * sizeof(double) + sizeof(size_t);

    const size_t split = pd.n / pd.nrank;
    const size_t remaining = pd.n - pd.nrank * split;
    const MPI_Offset col = pd.n * sizeof(double);

    MPI_Offset offset = headerOffset + pd.rank * split * sizeof(double);
    if (pd.rank > pd.nrank - 1) offset += remaining * sizeof(double);

    pd.count = pd.rank != pd.nrank - 1 ? split : split + remaining;
    pd.resize(pd.count);

    MPI_File_read(fh, &pd.ttot, 1, MPI_DOUBLE, &status);
    MPI_File_read(fh, &pd.minDt, 1, MPI_DOUBLE, &status);

    fileutils::details::readFileMPI(fh, pd.count, offset, col, 0, data...);

    MPI_File_close(&fh);
}
// }}}

} // namespace fileutils
} // namespace sphexa
