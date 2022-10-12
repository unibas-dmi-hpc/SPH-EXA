/*
 * MIT License
 *
 * Copyright (c) 2022 CSCS, ETH Zurich
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief file I/O interface
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <map>
#include <string>
#include <vector>

#include "cstone/sfc/box.hpp"

#include "file_utils.hpp"
#ifdef SPH_EXA_HAVE_H5PART
#include "mpi_file_utils.hpp"
#endif

namespace sphexa
{

template<typename Dataset>
struct IFileWriter
{
    virtual void dump(Dataset& d, size_t firstIndex, size_t lastIndex,
                      const cstone::Box<typename Dataset::RealType>& box, std::string path) const = 0;
    virtual void constants(const std::map<std::string, double>& c, std::string path) const        = 0;

    virtual std::string suffix() const = 0;

    virtual ~IFileWriter() = default;
};

template<class Dataset>
struct AsciiWriter : public IFileWriter<Dataset>
{
    void dump(Dataset& simData, size_t firstIndex, size_t lastIndex, const cstone::Box<typename Dataset::RealType>& box,
              std::string path) const override
    {
        auto&      d         = simData.hydro;
        const char separator = ' ';
        path += std::to_string(d.ttot) + ".txt";

        int rank, numRanks;
        MPI_Comm_rank(simData.comm, &rank);
        MPI_Comm_size(simData.comm, &numRanks);

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
                    MPI_Abort(simData.comm, 1);
                }
            }

            MPI_Barrier(simData.comm);
        }
    }

    void constants(const std::map<std::string, double>& c, std::string path) const override {}

    std::string suffix() const override { return ""; }
};

template<class Dataset>
struct H5PartWriter : public IFileWriter<Dataset>
{
    void dump(Dataset& simData, size_t firstIndex, size_t lastIndex, const cstone::Box<typename Dataset::RealType>& box,
              std::string path) const override
    {
        auto& d = simData.hydro;
#ifdef SPH_EXA_HAVE_H5PART
        fileutils::writeH5Part(d, firstIndex, lastIndex, box, path, simData.comm);
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
        const char* h5_fname = path.c_str();

        if (std::filesystem::exists(h5_fname))
        {
            throw std::runtime_error("Cannot write constants: file " + path + " already exists\n");
        }

        H5PartFile* h5_file = nullptr;
        h5_file             = H5PartOpenFile(h5_fname, H5PART_WRITE);

        for (auto it = c.begin(); it != c.end(); ++it)
        {
            fileutils::sphexaWriteFileAttrib(h5_file, it->first.c_str(), &(it->second), 1);
        }

        H5PartCloseFile(h5_file);
#else
        throw std::runtime_error("Cannot write to HDF5 file: H5Part not enabled\n");
#endif
    }

    std::string suffix() const override { return ".h5"; }
};

template<class Dataset>
std::unique_ptr<IFileWriter<Dataset>> fileWriterFactory(bool ascii)
{
    if (ascii) { return std::make_unique<AsciiWriter<Dataset>>(); }
    else { return std::make_unique<H5PartWriter<Dataset>>(); }
}

} // namespace sphexa
