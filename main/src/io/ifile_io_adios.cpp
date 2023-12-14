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
 * @brief File I/O interface implementation with ADIOS2
 *
 * @author Yiqing Zhu <yiqing.zhu@unibas.ch>
 */

#include <mpi.h>

#include <filesystem>
#include <string>
#include <variant>
#include <vector>

#include "ifile_io_impl.h"
#ifdef SPH_EXA_HAVE_ADIOS
#include "adios_wrapper.hpp"
#endif


namespace sphexa
{
#ifdef SPH_EXA_HAVE_ADIOS
class ADIOSWriter final : public IFileWriter
{
public:
    using Base      = IFileWriter;
    using FieldType = typename Base::FieldType;

    explicit ADIOSWriter(MPI_Comm comm, const std::string& compressionMethod, const std::string& compressionParam = "")
        : comm_(comm)
    {
        MPI_Comm_rank(comm, &rank_);
        as_.accuracy = 0.00001;
        as_.rank = rank_;
        as_.comm = comm;
        // if (compressionMethod == "gzip") h5z_.compression = fileutils::CompressionMethod::gzip;
        // if (compressionMethod == "szip") h5z_.compression = fileutils::CompressionMethod::szip;
        // if (compressionMethod == "zfp") h5z_.compression = fileutils::CompressionMethod::zfp;
        as_.accuracy = std::stof(compressionParam);
    }

    ~ADIOSWriter() override { closeStep(); }

    [[nodiscard]] int rank() const override { return rank_; }
    [[nodiscard]] int numRanks() const override { return numRanks_; }

    std::string suffix() const override { return ".bp"; }

    void addStep(size_t firstIndex, size_t lastIndex, std::string path) override
    {
        firstIndex_ = firstIndex;
        MPI_Comm_rank(comm_, &rank_);
        MPI_Comm_size(comm_, &totalRanks_);

        MPI_Barrier(MPI_COMM_WORLD);
        fileInitTime_ = -MPI_Wtime();

        // BP doesn't have hierarchical structure, thus each timestep
        // has a unique specifier in the variable name. When reading in,
        // we use regex for parsing the hierarchy.
        as_.comm = comm_;
        as_.fileName = path;
        // In BP we use a "Step#X_" prefix to identify steps
        if (lastIndex > firstIndex) {
            as_.numLocalParticles = lastIndex - firstIndex;
            currStep_         = currStep_ + 1;
            as_.stepPrefix     = "Step#" + std::to_string(currStep_ - 1) + "_";
        }
        else {
            as_.numLocalParticles = 0;
        }
        as_.numTotalRanks = totalRanks_;
        as_.offset = firstIndex;

        // For now, the writer will only append data instead of writing new
        fileutils::initADIOSWriter(as_);

        MPI_Barrier(MPI_COMM_WORLD);
        fileInitTime_ += MPI_Wtime();
        pathStep_ = path;
    }

    void stepAttribute(const std::string& key, FieldType val, int64_t size) override
    {
        std::visit([this, &key, size](auto arg) {
            fileutils::writeADIOSAttribute(as_, key, arg);
        }, val);
    }

    void fileAttribute(const std::string& key, FieldType val, int64_t size) override
    {
        std::visit([this, &key, size](auto arg) { 
            fileutils::writeADIOSAttribute(as_, key, arg);
         }, val);
    }

    void writeField(const std::string& key, FieldType field, int = 0) override
    {
        MPI_Barrier(MPI_COMM_WORLD);
        writeTime_ = -MPI_Wtime();

        // If there's a need to change particle numbers, do it here and now!!
        // Directly change it in as_.
        std::visit(
            [this, &key](auto arg)
            {
                fileutils::writeADIOSField(as_, key, arg);
            },
            field);

        MPI_Barrier(MPI_COMM_WORLD);
        writeTime_ += MPI_Wtime();
    }

    void setNumParticles(uint64_t numParticles) override { totalNumParticles_ = numParticles; }

    void setCompression(const std::string& compressionMethod, const std::string& compressionParam) override
    {
        // if (compressionMethod == "gzip") h5z_.compression = fileutils::CompressionMethod::gzip;
        // if (compressionMethod == "szip") h5z_.compression = fileutils::CompressionMethod::szip;
        // if (compressionMethod == "zfp") h5z_.compression = fileutils::CompressionMethod::zfp;
        // h5z_.compressionParam = compressionParam;
    }

    void closeStep() override
    {
        if (rank_ == 0)
        {
            std::cout << "Writter!!!File init elapse: " << fileInitTime_ << ", writing elapse: " << writeTime_ << std::endl;
        }
    }

private:
    int      rank_{0}, numRanks_{0};
    int      totalRanks_{0};
    size_t   totalNumParticles_{0};
    double   fileInitTime_, writeTime_;
    MPI_Comm comm_;

    size_t      firstIndex_{0};
    std::string pathStep_;
    size_t      currStep_{0};

    fileutils::ADIOS2Settings as_;
};

std::unique_ptr<IFileWriter> makeADIOSWriter(MPI_Comm comm, const std::string& compressionMethod,
                                               const std::string& compressionParam) {
        return std::make_unique<ADIOSWriter>(comm, compressionMethod, compressionParam);
}

#else

std::unique_ptr<IFileWriter> makeADIOSWriter(MPI_Comm, const std::string&,
                                               const std::string&) { return {}; }

#endif

} // namespace sphexa