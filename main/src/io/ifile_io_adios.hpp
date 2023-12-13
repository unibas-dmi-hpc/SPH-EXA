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

#pragma once

#include <mpi.h>

#include <filesystem>
#include <string>
#include <variant>
#include <vector>

#include "init/grid.hpp"

#include "mpi_file_utils.hpp"
#include "h5part_wrapper.hpp"
#include "adios_wrapper.hpp"
#include "ifile_io.hpp"

namespace sphexa
{

class ADIOSWriter final : public IFileWriter
{
public:
    using Base      = IFileWriter;
    using FieldType = typename Base::FieldType;

    explicit ADIOSWriter(MPI_Comm comm, const std::string& compressionMethod, const int& compressionParam = 0)
        : comm_(comm)
    {
        MPI_Comm_rank(comm, &rank_);
        as_.accuracy = 0.00001;
        as_.rank = rank_;
        as_.comm = comm;

    }

    ~ADIOSWriter() override { closeStep(); }

    [[nodiscard]] int rank() const override { return rank_; }

    std::string suffix() const override { return ".h5"; }

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
        // In BP we use a "Step#0_" prefix to identify steps
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

        // if ((!h5z_.file_id) || (path != pathStep_))
        // {
        //     if (std::filesystem::exists(path)) { 
        //         fileutils::openHDF5File(path, comm_); }
        //     else {  }
                
        // }

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
        // long long int currIndex[totalRanks_ + 1];
        // currIndex[0] = 0;
        // int ret = MPI_Allgather((void*)&h5z_.numParticles, 1, MPI_LONG_LONG, (void*)&currIndex[1], 1, MPI_LONG_LONG,
        //                         MPI_COMM_WORLD);
        // if (ret != MPI_SUCCESS) { std::cout << "error! " << std::endl; };
        // for (int i = 1; i < totalRanks_ + 1; i++)
        // {
        //     currIndex[i] += currIndex[i - 1];
        // }

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

    void setCompression(const std::string& compressionMethod, int compressionParam) override
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
        // fileutils::closeHDF5File(h5z_);
    }

private:
    int      rank_{0};
    int      totalRanks_{0};
    size_t   totalNumParticles_{0};
    double   fileInitTime_, writeTime_;
    MPI_Comm comm_;

    size_t      firstIndex_{0};
    std::string pathStep_;
    size_t      currStep_{0};

    fileutils::compressionSettings cs_;
    fileutils::ADIOS2Settings as_;
};

class ADIOSReader final : public IFileReader
{
public:
    using Base      = IFileReader;
    using FieldType = typename Base::FieldType;

    explicit ADIOSReader(MPI_Comm comm)
        : comm_(comm)
        , h5File_{nullptr}
    {
        MPI_Comm_rank(comm, &rank_);
    }

    ~ADIOSReader() override { closeStep(); }

    [[nodiscard]] int     rank() const override { return rank_; }
    [[nodiscard]] int64_t numParticles() const override
    {
        if (!h5File_) { throw std::runtime_error("Cannot get number of particles: file not open\n"); }
        return H5PartGetNumParticles(h5File_);
    }

    /*! @brief open a file at a given step
     *
     * @param path  filesystem path
     * @param step  snapshot index to load from
     * @param mode  collective mode causes MPI ranks to distribute particles amongst themselves,
     *              independent mode causes all MPI ranks to load all particles
     */
    void setStep(std::string path, int step, FileMode mode) override
    {
        closeStep();
        pathStep_ = path;
        h5File_   = fileutils::openH5Part(path, H5PART_READ | H5PART_VFD_MPIIO_IND, comm_);

        size_t fileNumSteps = H5PartGetNumSteps(h5File_);

        if (fileNumSteps == 0) { return; }

        // set step to last step in file if negative
        std::cout << "Current steps: " << step << ", file steps: " << fileNumSteps << std::endl;
        if (step < 0) { step = fileNumSteps - 1; }
        H5PartSetStep(h5File_, step);

        globalCount_ = H5PartGetNumParticles(h5File_);
        if (globalCount_ < 1) { return; }

        int rank, numRanks;
        MPI_Comm_rank(comm_, &rank);
        MPI_Comm_size(comm_, &numRanks);

        if (mode == FileMode::collective)
        {
            std::tie(firstIndex_, lastIndex_) = partitionRange(globalCount_, rank, numRanks);
            localCount_                       = lastIndex_ - firstIndex_;
            h5z_              = fileutils::openHDF5File(path, comm_);
            h5z_.step         = step;
            h5z_.start        = firstIndex_;
            h5z_.numParticles = localCount_;
            H5PartSetView(h5File_, firstIndex_, lastIndex_ - 1);
        }
        else { std::tie(firstIndex_, lastIndex_, localCount_) = std::make_tuple(0, globalCount_, globalCount_); }
    }

    std::vector<std::string> fileAttributes() override
    {
        if (h5File_) { return fileutils::fileAttributeNames(h5File_); }
        else { throw std::runtime_error("Cannot read file attributes: file not opened\n"); }
    }

    std::vector<std::string> stepAttributes() override
    {
        if (h5File_) { return fileutils::stepAttributeNames(h5File_); }
        else { throw std::runtime_error("Cannot read file attributes: file not opened\n"); }
    }

    int64_t fileAttributeSize(const std::string& key) override
    {
        int64_t attrIndex = fileAttributeIndex(key);
        int64_t typeId, attrSize;
        char    dummy[256];
        H5PartGetFileAttribInfo(h5File_, attrIndex, dummy, 256, &typeId, &attrSize);
        return attrSize;
    }

    int64_t stepAttributeSize(const std::string& key) override
    {
        int64_t attrIndex = stepAttributeIndex(key);
        int64_t typeId, attrSize;
        char    dummy[256];
        H5PartGetStepAttribInfo(h5File_, attrIndex, dummy, 256, &typeId, &attrSize);
        return attrSize;
    }

    void fileAttribute(const std::string& key, FieldType val, int64_t size) override
    {
        std::visit(
            [this, size, &key](auto arg)
            {
                auto index = fileAttributeIndex(key);
                fileutils::readH5PartFileAttribute(arg, size, index, h5File_);
            },
            val);
    }

    void stepAttribute(const std::string& key, FieldType val, int64_t size) override
    {
        std::visit(
            [this, size, &key](auto arg)
            {
                auto index = stepAttributeIndex(key);
                fileutils::readH5PartStepAttribute(arg, size, index, h5File_);
            },
            val);
    }

    void readField(const std::string& key, FieldType field) override
    {
        // auto err = std::visit([this, &key](auto arg) { return fileutils::readH5PartField(h5File_, key, arg); }, field);
        // if (err != H5PART_SUCCESS) { throw std::runtime_error("Could not read field: " + key); }

        if (rank_ == 0)
        {
            std::cout << "Start reading in rank " << rank_ << "global:" << globalCount_ << std::endl;
            auto err = std::visit([this, &key](auto arg) { return fileutils::readHDF5Field(h5z_, key, arg, globalCount_); },
                        field);
            if (err != H5PART_SUCCESS) { throw std::runtime_error("Could not read field: " + key); }
        }

        
    }

    uint64_t localNumParticles() override { return localCount_; }

    uint64_t globalNumParticles() override { return globalCount_; }

    void closeStep() override
    {
        // if (h5File_)
        // {
        //     H5PartCloseFile(h5File_);
        //     h5File_ = nullptr;
        // }

        if (rank_ == 0)
        {
            std::cout << "File init elapse: " << fileInitTime_ << ", writing elapse: " << writeTime_ << std::endl;
        }
        fileutils::closeHDF5File(h5z_, true);
    }

private:
    int64_t stepAttributeIndex(const std::string& key)
    {
        auto    attributes = fileutils::stepAttributeNames(h5File_);
        int64_t attrIndex  = std::find(attributes.begin(), attributes.end(), key) - attributes.begin();
        if (attrIndex == attributes.size()) { throw std::out_of_range("Attribute " + key + " does not exist\n"); }
        return attrIndex;
    }

    int64_t fileAttributeIndex(const std::string& key)
    {
        auto    attributes = fileutils::fileAttributeNames(h5File_);
        int64_t attrIndex  = std::find(attributes.begin(), attributes.end(), key) - attributes.begin();
        if (attrIndex == attributes.size()) { throw std::out_of_range("Attribute " + key + " does not exist\n"); }
        return attrIndex;
    }

    int      rank_{0};
    MPI_Comm comm_;

    uint64_t    firstIndex_, lastIndex_;
    uint64_t    localCount_;
    uint64_t    globalCount_;
    std::string pathStep_;

    H5PartFile* h5File_;
    fileutils::H5ZType h5z_;
    double      fileInitTime_, writeTime_;
};

} // namespace sphexa