#pragma once

#include <memory>
#include <mpi.h>

#include "ifile_io.hpp"

namespace sphexa
{

std::unique_ptr<IFileWriter> makeAsciiWriter(MPI_Comm comm);
std::unique_ptr<IFileWriter> makeH5PartWriter(MPI_Comm comm);
std::unique_ptr<IFileWriter> makeADIOSWriter(MPI_Comm comm, const std::string& compressionMethod,
                                             const std::string& compressionParam = "");

std::unique_ptr<IFileReader> makeH5PartReader(MPI_Comm comm);
std::unique_ptr<IFileReader> makeADIOSReader(MPI_Comm comm, const std::string& compressionMethod,
                                             const std::string& compressionParam = "");

} // namespace sphexa
