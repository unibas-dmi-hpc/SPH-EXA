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
    H5PartFile* h5_file  = nullptr;
    // open file
    if (std::filesystem::exists(h5_fname)) { h5_file = H5PartOpenFile(h5_fname, H5PART_APPEND); }
    else
    {
        h5_file = H5PartOpenFile(h5_fname, H5PART_WRITE);
    }

    // create first step
    const h5_id_t h5_step = d.iteration;
    H5PartSetStep(h5_file, h5_step);

    // get number of particles that each rank will write
    const int        h5_begin         = firstIndex;
    const int        h5_end           = lastIndex;
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
    for (auto ii = h5_begin; ii < h5_end; ii++)
    {
        auto jj        = ii - h5_begin;
        h5_data_x[jj]  = d.x[ii];
        h5_data_y[jj]  = d.y[ii];
        h5_data_z[jj]  = d.z[ii];
        h5_data_h[jj]  = d.h[ii];
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

} // namespace fileutils
} // namespace sphexa
