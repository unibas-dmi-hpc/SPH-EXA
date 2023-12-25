/*
 * MIT License
 *
 * Copyright (c) 2023 CSCS, ETH Zurich, University of Basel, University of Zurich
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

#include <memory>
#include <filesystem>

#include "ifile_io_impl.h"

namespace sphexa
{

std::unique_ptr<IFileWriter> fileWriterFactory(bool ascii, MPI_Comm comm, const std::string& filePath = "",
                                               const std::string& compressionMethod = "",
                                               const std::string& compressionParam  = "")
{
    if (ascii) { return makeAsciiWriter(comm); }
    // If the file suffix ends in ".bp", use ADIOS reader/writer.
    // else use H5Part reader/writer, but then compression is unavailable.
    auto suffix = std::filesystem::path(filePath).extension().string();
    if (suffix == ".bp")
    {
#ifdef SPH_EXA_HAVE_ADIOS
        return makeADIOSWriter(comm, compressionMethod, compressionParam);
#endif
        throw std::runtime_error(
            "unsupported compression file i/o choice. BP I/O is only available with ADIOS2 enabled.\n");
    }
    else
    {
        if (compressionParam != "" || compressionMethod != "")
        {
            throw std::runtime_error("unsupported compression file i/o choice. Output compression is only available "
                                     "with BP file and ADIOS2 enabled.\n");
        }
        return makeH5PartWriter(comm);
    }
}

std::unique_ptr<IFileReader> fileReaderFactory(bool /*ascii*/, MPI_Comm comm, const std::string& filePath = "",
                                               const std::string& compressionMethod = "",
                                               const std::string& compressionParam  = "")
{
    auto suffix = std::filesystem::path(filePath).extension().string();
    if (suffix == ".bp")
    {
#ifdef SPH_EXA_HAVE_ADIOS
        return makeADIOSReader(comm, compressionMethod, compressionParam);
#endif
        throw std::runtime_error("unsupported file i/o choice. BP I/O is only available with ADIOS2 enabled.\n");
    }
    return makeH5PartReader(comm);
}

} // namespace sphexa