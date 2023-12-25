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

#include "ifile_io_impl.h"

namespace sphexa
{

std::unique_ptr<IFileWriter> fileWriterFactory(bool ascii, MPI_Comm comm, const std::string& compressionMethod,
                                               const std::string& compressionParam = "")
{
    if (ascii) { return makeAsciiWriter(comm); }
    else
    {
        if (compressionMethod == "")
        {
// If no compression specified, both adios and h5part can work.
// Adios is preferred for compatibility reasons
#ifdef SPH_EXA_HAVE_ADIOS
            return makeADIOSWriter(comm, compressionMethod, compressionParam);
#endif
            return makeH5PartWriter(comm);
        }
        else
        {
// If compression, only adios would work
#ifdef SPH_EXA_HAVE_ADIOS
            return makeADIOSWriter(comm, compressionMethod, compressionParam);
#endif
            throw std::runtime_error(
                "unsupported compression file i/o choice. Compression is only available with ADIOS.\n");
        }
    }
}

std::unique_ptr<IFileReader> fileReaderFactory(bool /*ascii*/, MPI_Comm comm, const std::string& compressionMethod,
                                               const std::string& compressionParam = "")
{

#ifdef SPH_EXA_HAVE_ADIOS
    return makeADIOSReader(comm, compressionMethod, compressionParam);
#endif
    return makeH5PartReader(comm);
}

} // namespace sphexa