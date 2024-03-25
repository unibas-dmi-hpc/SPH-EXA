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

    explicit ADIOSWriter(MPI_Comm comm, const std::string& compression = "")
        : comm_(comm)
        , as_(compression)
    {
        MPI_Comm_rank(comm, &rank_);
        as_.rank = rank_;
        as_.comm = comm;
    }

    ~ADIOSWriter() override {}

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

        // Here it's mandatory to refresh rank num into as_
        if (lastIndex > firstIndex)
        {
            as_.numLocalParticles = lastIndex - firstIndex;
            currStep_             = currStep_ + 1;
        }
        else { as_.numLocalParticles = 0; }
        as_.comm          = MPI_COMM_WORLD;
        as_.fileName      = path;
        as_.rank          = rank_;
        as_.numTotalRanks = totalRanks_;
        as_.offset        = firstIndex;
        as_.currStep      = currStep_;

        long long int currIndex[totalRanks_ + 1];
        currIndex[0] = 0;
        int ret = MPI_Allgather((void*)&as_.numLocalParticles, 1, MPI_LONG_LONG, (void*)&currIndex[1], 1, MPI_LONG_LONG,
                                MPI_COMM_WORLD);
        if (ret != MPI_SUCCESS) { std::cout << "error! " << std::endl; };

        as_.numGlobalParticles = 0;
        for (int i = 1; i < totalRanks_ + 1; i++)
        {
            as_.numGlobalParticles += currIndex[i];
            currIndex[i] += currIndex[i - 1];
        }
        as_.offset = currIndex[as_.rank];

        // std::cout << "rank: " << as_.rank << ", start: " << firstIndex << ", end:" << lastIndex << endl;

        // For now, the writer will only append data instead of writing new
        fileutils::initADIOSWriter(as_);
        fileutils::openADIOSStepWrite(as_);

        MPI_Barrier(MPI_COMM_WORLD);
        fileInitTime_ += MPI_Wtime();
        pathStep_  = path;
        writeTime_ = 0;
    }

    void stepAttribute(const std::string& key, FieldType val, int64_t size) override
    {
        MPI_Barrier(MPI_COMM_WORLD);
        writeTime_ += -MPI_Wtime();
        if (key == "rngEngineState") { return fileAttribute(key, val, size); }
        std::visit([this, &key, size](auto arg) { fileutils::writeADIOSStepAttribute(as_, key, arg, size); }, val);
        MPI_Barrier(MPI_COMM_WORLD);
        writeTime_ += MPI_Wtime();
    }

    void fileAttribute(const std::string& key, FieldType val, int64_t size) override
    {
        std::visit([this, &key, size](auto arg) { fileutils::writeADIOSFileAttribute(as_, key, arg, size); }, val);
    }

    void writeField(const std::string& key, FieldType field, int = 0) override
    {
        MPI_Barrier(MPI_COMM_WORLD);
        writeTime_ += -MPI_Wtime();

        // If there's a need to change particle numbers, do it here and now!!
        // Directly change it in as_.
        std::visit([this, &key](auto arg) { fileutils::writeADIOSField(as_, key, arg + firstIndex_); }, field);

        MPI_Barrier(MPI_COMM_WORLD);
        writeTime_ += MPI_Wtime();
    }

    void closeStep() override
    {
        fileutils::closeADIOSStepWrite(as_);
        if (rank_ == 0)
        {
            std::cout << "ADIOS2 Writer -- File init elapse: " << fileInitTime_ << ", writing elapse: " << writeTime_
                      << std::endl;
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

std::unique_ptr<IFileWriter> makeADIOSWriter(MPI_Comm comm, const std::string& compression)
{
    return std::make_unique<ADIOSWriter>(comm, compression);
}

inline auto partitionRange(size_t R, size_t i, size_t N)
{
    size_t s = R / N;
    size_t r = R % N;
    if (i < r)
    {
        size_t start = (s + 1) * i;
        size_t end   = start + s + 1;
        return std::make_tuple(start, end);
    }
    else
    {
        size_t start = (s + 1) * r + s * (i - r);
        size_t end   = start + s;
        return std::make_tuple(start, end);
    }
}

class ADIOSReader final : public IFileReader
{
public:
    using Base      = IFileReader;
    using FieldType = typename Base::FieldType;

    explicit ADIOSReader(MPI_Comm comm, const std::string& compression = "")
        : comm_(comm)
        , as_(compression)
    {
        MPI_Comm_rank(comm, &rank_);
        as_.rank = rank_;
        as_.comm = comm;
    }

    ~ADIOSReader() override {}

    [[nodiscard]] int     rank() const override { return rank_; }
    [[nodiscard]] int64_t numParticles() const override { return globalCount_; }

    /*! @brief open a file at a given step
     *
     * @param path  filesystem path
     * @param step  snapshot index to load from
     * @param mode  collective mode causes MPI ranks to distribute particles amongst themselves,
     *              independent mode causes all MPI ranks to load all particles
     */
    void setStep(std::string path, int step, FileMode mode) override
    {
        pathStep_    = path;
        as_.comm     = comm_;
        as_.fileName = path;
        MPI_Barrier(MPI_COMM_WORLD);
        fileInitTime_ = -MPI_Wtime();
        fileutils::initADIOSReader(as_);
        fileutils::openADIOSStepRead(as_);

        int64_t totalSteps = fileutils::ADIOSGetNumSteps(as_);

        // Step should >= 1
        if (step <= totalSteps && step > 0) { as_.currStep = step; }
        as_.setCompressor(fileutils::readADIOSFileCompressorSettings(as_));

        // set step to last iteration in file if negative
        if (step < 0)
        {
            as_.currStep = totalSteps;
            step         = ADIOSGetNumIterations(as_);
        }

        globalCount_ = fileutils::ADIOSGetNumParticles(as_);
        if (globalCount_ < 1) { return; }

        int rank, numRanks;
        MPI_Comm_rank(comm_, &rank);
        MPI_Comm_size(comm_, &numRanks);

        if (mode == FileMode::collective)
        {
            std::tie(firstIndex_, lastIndex_) = partitionRange(globalCount_, rank, numRanks);
            localCount_                       = lastIndex_ - firstIndex_;
        }
        else { std::tie(firstIndex_, lastIndex_, localCount_) = std::make_tuple(0, globalCount_, globalCount_); }

        as_.numLocalParticles = localCount_;
        as_.numTotalRanks     = numRanks;
        as_.offset            = firstIndex_;
        as_.rank              = rank;

        long long int currIndex[numRanks + 1];
        currIndex[0] = 0;
        int ret = MPI_Allgather((void*)&as_.numLocalParticles, 1, MPI_LONG_LONG, (void*)&currIndex[1], 1, MPI_LONG_LONG,
                                MPI_COMM_WORLD);
        if (ret != MPI_SUCCESS) { std::cout << "error! " << std::endl; };
        as_.numGlobalParticles = 0;
        for (int i = 1; i < numRanks + 1; i++)
        {
            as_.numGlobalParticles += currIndex[i];
            currIndex[i] += currIndex[i - 1];
        }

        as_.offset = currIndex[as_.rank];
        std::cout << "read: offset:" << as_.offset << ", rank: " << as_.offset
                  << ", numLocalParticles: " << as_.numLocalParticles << std::endl;

        MPI_Barrier(MPI_COMM_WORLD);
        fileInitTime_ += MPI_Wtime();
        readTime_ = 0;
    }

    std::vector<std::string> fileAttributes() override { return ADIOSGetFileAttributes(as_); }

    std::vector<std::string> stepAttributes() override { return std::vector<std::string>(); }

    int64_t fileAttributeSize(const std::string& key) override { return ADIOSGetFileAttributeSize(as_, key); }

    int64_t stepAttributeSize(const std::string& key) override { return ADIOSGetStepAttributeSize(as_, key); }

    void fileAttribute(const std::string& key, FieldType val, int64_t size) override
    {
        MPI_Barrier(MPI_COMM_WORLD);
        readTime_ += -MPI_Wtime();
        std::visit([this, size, &key](auto arg) { fileutils::readADIOSFileAttribute(as_, key, arg, size); }, val);
        MPI_Barrier(MPI_COMM_WORLD);
        readTime_ += MPI_Wtime();
    }

    void stepAttribute(const std::string& key, FieldType val, int64_t size) override
    {
        MPI_Barrier(MPI_COMM_WORLD);
        readTime_ += -MPI_Wtime();
        try
        {
            std::visit([this, size, &key](auto arg) { fileutils::readADIOSStepAttribute(as_, key, arg, size); }, val);
        }
        catch (const std::invalid_argument& e)
        {
            return fileAttribute(key, val, size);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        readTime_ += MPI_Wtime();
    }

    void readField(const std::string& key, FieldType field) override
    {
        MPI_Barrier(MPI_COMM_WORLD);
        readTime_ += -MPI_Wtime();
        std::visit([this, &key](auto arg) { fileutils::readADIOSField(as_, key, arg); }, field);
        // std::visit(
        //     [this, &key](auto arg)
        //     {
        //         if (key == "x")
        //         {
        //             for (int i = 0; i < 125000; i++)
        //             {
        //                 std::cout << arg[i] << ",";
        //             }
        //             std::cout << std::endl;
        //         }
        //     },
        //     field);
        MPI_Barrier(MPI_COMM_WORLD);
        readTime_ += MPI_Wtime();
    }

    uint64_t localNumParticles() override { return localCount_; }

    uint64_t globalNumParticles() override { return globalCount_; }

    void closeStep() override
    {
        if (rank_ == 0)
        {
            std::cout << "ADIOS2 Reader -- File init elapse: " << fileInitTime_ << ", reading elapse: " << readTime_
                      << std::endl;
        }

        closeADIOSStepRead(as_);
    }

private:
    int      rank_{0}, numRanks_{0};
    MPI_Comm comm_;

    double fileInitTime_, readTime_;

    uint64_t    firstIndex_{0}, lastIndex_{0};
    uint64_t    localCount_;
    uint64_t    globalCount_;
    std::string pathStep_;

    fileutils::ADIOS2Settings as_;
};

std::unique_ptr<IFileReader> makeADIOSReader(MPI_Comm comm, const std::string& compression)
{
    return std::make_unique<ADIOSReader>(comm, compression);
}

#else

std::unique_ptr<IFileWriter> makeADIOSWriter(MPI_Comm, const std::string&) { return {}; }
std::unique_ptr<IFileReader> makeADIOSReader(MPI_Comm, const std::string&) { return {}; }

#endif

} // namespace sphexa
