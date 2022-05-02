#pragma once

#include <fstream>
#include <vector>
#include <variant>

namespace sphexa
{
namespace fileutils
{

//! @brief write a single line of compile-time fixed column types to an ostream
template<class Separator, class... Columns>
void writeColumns(std::ostream& out, const Separator& sep, Columns&&... columns)
{
    [[maybe_unused]] std::initializer_list<int> list{(out << sep << columns, 0)...};
    out << std::endl;
}

/*! @brief write fields as columns to an ASCII file
 *
 * @tparam  T              field value type
 * @tparam  Separators
 * @param   firstIndex     first field index to write
 * @param   lastIndex      last field index to write
 * @param   path           the file name to write to
 * @param   append         append or overwrite if file already exists
 * @param   fields         pointers to field array, each field is a column
 * @param   separators     arbitrary number of separators to insert between columns, eg '\t', std::setw(n), ...
 */
template<class... T, class... Separators>
void writeAscii(size_t firstIndex, size_t lastIndex, const std::string& path, bool append,
                const std::vector<std::variant<T*...>>& fields, Separators&&... separators)
{
    std::ios_base::openmode mode;
    if (append) { mode = std::ofstream::app; }
    else { mode = std::ofstream::out; }

    std::ofstream dumpFile(path, mode);

    if (dumpFile.is_open())
    {
        for (size_t i = firstIndex; i < lastIndex; ++i)
        {
            for (auto field : fields)
            {
                [[maybe_unused]] std::initializer_list<int> list{(dumpFile << separators, 0)...};
                std::visit([&dumpFile, i](auto& arg) { dumpFile << arg[i]; }, field);
            }
            dumpFile << std::endl;
        }
    }
    else { throw std::runtime_error("Can't open file at path: " + path); }

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
    else { throw std::runtime_error("Can't open file at path: " + path); }
}

} // namespace fileutils
} // namespace sphexa
