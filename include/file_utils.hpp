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

template<class T>
void writeAscii(size_t firstIndex, size_t lastIndex, const std::string& path, bool append, char separator,
                const std::vector<T*>& fields)
{
    std::ios_base::openmode mode;
    if (append) { mode = std::ofstream::app; }
    else
    {
        mode = std::ofstream::out;
    }

    std::ofstream dumpFile(path, mode);

    if (dumpFile.is_open())
    {
        for (size_t i = firstIndex; i < lastIndex; ++i)
        {
            for (auto field : fields)
            {
                dumpFile << field[i] << separator;
            }
            dumpFile << std::endl;
        }
    }
    else
    {
        throw FileNotOpenedException("Can't open file at path: " + path);
    }

    dumpFile.close();
}

/*! @brief read input data from an ASCII file
 *
 * @tparam T         an elementary type or a std::variant thereof
 * @param  path      the input file to read from
 * @param  numLines  number of lines/elements per field to read
 * @param  fields    the data containers to read into
 *
 *  Each data container will get one column of the input file.
 *  The number of rows to read is determined by the data container size.
 */
template<class T>
void readAscii(const std::string& path, size_t numLines, const std::vector<T*>& fields)
{
    std::ifstream inputFile(path);

    if (inputFile.is_open())
    {
        for (size_t i = 0; i < numLines; ++i)
        {
            for (auto field : fields)
            {
                inputFile >> field[i];
            }
        }
        inputFile.close();
    }
    else
    {
        throw FileNotOpenedException("Can't open file at path: " + path);
    }
}

} // namespace fileutils
} // namespace sphexa
