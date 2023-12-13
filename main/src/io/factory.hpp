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

#include "ifile_io.hpp"
#include "ifile_io_ascii.hpp"
#ifdef SPH_EXA_HAVE_HDF5
#include "ifile_io_hdf5.hpp"
#endif
#ifdef SPH_EXA_HAVE_H5PART
#include "ifile_io_h5part.hpp"
#endif
#ifdef SPH_EXA_HAVE_ADIOS
#include "ifile_io_adios.hpp"
#endif

namespace sphexa
{

std::unique_ptr<IFileWriter> fileWriterFactory(bool ascii, MPI_Comm comm, const std::string& compressionMethod,
                                               const int& compressionParam = 0)
{
    if (ascii) { return std::make_unique<AsciiWriter>(comm); }
    if (compressionMethod == "") {
#ifdef SPH_EXA_HAVE_H5PART
        return std::make_unique<H5PartWriter>(comm);
#endif
        throw std::runtime_error("unsupported file i/o choice\n");
    } else {
        // If ADIOS is available, use ADIOS
        // Otherwise if user didn't specify compression method at all, use H5Part
#ifdef SPH_EXA_HAVE_ADIOS
        return std::make_unique<ADIOSWriter>(comm, compressionMethod, compressionParam);
#endif
        return std::make_unique<HDF5Writer>(comm, compressionMethod, compressionParam);
    }
}

std::unique_ptr<IFileReader> fileReaderFactory(bool /*ascii*/, MPI_Comm comm)
{
#if defined(SPH_EXA_HAVE_HDF5)
    return std::make_unique<HDF5Reader>(comm);
#elif defined(SPH_EXA_HAVE_H5PART)
    return std::make_unique<H5PartReader>(comm);
#else
    return std::make_unique<UnimplementedReader>();
#endif
}

} // namespace sphexa