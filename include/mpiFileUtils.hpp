#include <mpi.h>

#pragma once

namespace sphexa
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

} // namespace sphexa
