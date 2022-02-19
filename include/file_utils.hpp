#pragma once

#include <fstream>
#include <vector>

#ifdef USE_MPI
#include "mpi_file_utils.hpp"
#endif
#include "exceptions.hpp"

namespace sphexa
{
namespace fileutils
{
namespace details
{

void writeParticleDataToAsciiFile(std::ostream&, size_t, char) {}

template<typename Arg, typename... Args>
void writeParticleDataToAsciiFile(std::ostream& file, size_t idx, char separator, const Arg& first, const Args&... data)
{
    file << first[idx] << separator;

    writeParticleDataToAsciiFile(file, idx, separator, data...);
}

void readParticleDataFromAsciiFile(std::ifstream&, size_t) {}

template<typename Arg, typename... Args>
void readParticleDataFromAsciiFile(std::ifstream& file, size_t idx, Arg& first, Args&... data)
{
    file >> first[idx];

    readParticleDataFromAsciiFile(file, idx, data...);
}

} // namespace details

template<typename... Args>
void writeParticleDataToAsciiFile(size_t firstIndex, size_t lastIndex, const std::string& path, const bool append,
                                  const char separator, Args&... data)
{
    std::ios_base::openmode mode;
    if (append)
        mode = std::ofstream::app;
    else
        mode = std::ofstream::out;

    std::ofstream dump(path, mode);

    if (dump.is_open())
    {
        for (size_t i = firstIndex; i < lastIndex; ++i)
        {
            details::writeParticleDataToAsciiFile(dump, i, separator, data...);
            dump << std::endl;
        }
    }
    else
    {
        throw FileNotOpenedException("Can't open file at path: " + path);
    }

    dump.close();
}

/*! @brief read input data from an ASCII file
 *
 * @tparam Args  variadic number of vector-like objects
 * @param  path  the input file to read from
 * @param  data  the data containers to read into
 *               each data container needs to be the same size (checked)
 *
 *  Each data container will get one column of the input file.
 *  The number of rows to read is determined by the data container size.
 */
template<typename... Args>
void readParticleDataFromAsciiFile(const std::string& path, Args&... data)
{
    std::ifstream inputfile(path);

    std::array<size_t, sizeof...(Args)> sizes{data.size()...};
    size_t                              readSize = sizes[0];

    if (std::count(sizes.begin(), sizes.end(), readSize) != sizeof...(Args))
    {
        throw std::runtime_error("Argument vector sizes to read into are not equal\n");
    }

    if (inputfile.is_open())
    {
        for (size_t i = 0; i < readSize; ++i)
        {
            details::readParticleDataFromAsciiFile(inputfile, i, data...);
        }
        inputfile.close();
    }
    else
    {
        throw FileNotOpenedException("Can't open file at path: " + path);
    }
}

} // namespace fileutils
} // namespace sphexa
