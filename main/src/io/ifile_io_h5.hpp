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
 * @brief file I/O interface based on HDF5 and various plugins
 *
 * @author Yiqing Zhu <yiqing.zhu@unibas.ch>
 */

#pragma once

#include <mpi.h>

#include <filesystem>
#include <map>
#include <string>
#include <variant>
#include <vector>

#include "init/grid.hpp"

#include "file_utils.hpp"
#include "mpi_file_utils.hpp"
#include "ifile_io.hpp"

namespace sphexa
{
class HDF5Writer : public IFileWriter
{
public:
    using Base      = IFileWriter;
    using FieldType = typename Base::FieldType;

    explicit HDF5Writer(MPI_Comm comm, const std::string& compressionMethod, const int& compressionParam = 0)
        : comm_(comm)
        , h5File_(nullptr)
    {
        if (compressionMethod == "gzip") h5z_.compression = fileutils::CompressionMethod::gzip;
        if (compressionMethod == "szip") h5z_.compression = fileutils::CompressionMethod::szip;
        if (compressionMethod == "zfp") h5z_.compression = fileutils::CompressionMethod::zfp;
        h5z_.compressionParam = compressionParam;
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
    }

    std::string suffix() const override { return ".h5"; }

    void addStep(size_t firstIndex, size_t lastIndex, std::string path) override
    {
        firstIndex_ = firstIndex;
        lastIndex_  = lastIndex;
        pathStep_   = path;

        MPI_Comm_rank(comm_, &rank_);
        MPI_Comm_size(comm_, &totalRanks_);

        MPI_Barrier(MPI_COMM_WORLD);
        fileInitTime_ = -MPI_Wtime();
        if (std::filesystem::exists(path)) { h5z_ = fileutils::openHDF5File(path, comm_); }
        else { h5z_ = fileutils::createHDF5File(path, comm_); }

        h5z_.numParticles = lastIndex - firstIndex;
        currStep_         = currStep_ + 1;
        pathStep_         = "Step#" + std::to_string(currStep_ - 1);

        addHDF5Step(h5z_, pathStep_);

        MPI_Barrier(MPI_COMM_WORLD);
        fileInitTime_ += MPI_Wtime();

        return;
    }

    void stepAttribute(const std::string& key, FieldType val, int64_t size) override
    {
        std::visit([this, &key, size](auto arg) { fileutils::writeHDF5Attrib(h5z_, key.c_str(), arg, size); }, val);
    }

    void writeField(const std::string& key, FieldType field, int = 0) override
    {
        long long int currIndex[totalRanks_ + 1];
        currIndex[0] = 0;
        int ret = MPI_Allgather((void*)&h5z_.numParticles, 1, MPI_LONG_LONG, (void*)&currIndex[1], 1, MPI_LONG_LONG,
                                MPI_COMM_WORLD);
        if (ret != MPI_SUCCESS) { std::cout << "error! " << std::endl; };
        for (int i = 1; i < totalRanks_ + 1; i++)
        {
            currIndex[i] += currIndex[i - 1];
        }

        MPI_Barrier(MPI_COMM_WORLD);
        writeTime_ = -MPI_Wtime();
        std::visit(
            [this, &key, &currIndex](auto arg)
            {
                fileutils::writeHDF5Field(h5z_, key, arg + firstIndex_, currIndex[rank_], currIndex[rank_ + 1],
                                          totalNumParticles_);
            },
            field);
        MPI_Barrier(MPI_COMM_WORLD);
        writeTime_ += MPI_Wtime();
    }

    void setNumParticles(uint64_t numParticles) override { totalNumParticles_ = numParticles; }

    void setCompression(const std::string& compressionMethod, int compressionParam) override
    {
        if (compressionMethod == "gzip") h5z_.compression = fileutils::CompressionMethod::gzip;
        if (compressionMethod == "szip") h5z_.compression = fileutils::CompressionMethod::szip;
        if (compressionMethod == "zfp") h5z_.compression = fileutils::CompressionMethod::zfp;
        h5z_.compressionParam = compressionParam;
    }

    void closeStep() override
    {
        if (rank_ == 0)
        {
            std::cout << "File init elapse: " << fileInitTime_ << ", writing elapse: " << writeTime_ << std::endl;
        }
        fileutils::closeHDF5File(h5z_);
    }

private:
    MPI_Comm comm_;
    int      rank_;
    int      totalRanks_;
    size_t   totalNumParticles_;
    double   fileInitTime_, writeTime_;

    size_t      firstIndex_, lastIndex_;
    std::string pathStep_;
    size_t      currStep_ = 0;

    H5PartFile*        h5File_;
    fileutils::H5ZType h5z_;
};

class HDF5Reader : public IFileReader
{
public:
    using Base      = IFileReader;
    using FieldType = typename Base::FieldType;

    explicit HDF5Reader(MPI_Comm comm)
        : comm_(comm)
        , h5File_{nullptr}
    {
    }

    void setStep(std::string path, int step) override
    {
        if (h5File_) { closeStep(); }
        pathStep_ = path;
        h5File_   = fileutils::openH5Part(path, H5PART_READ | H5PART_VFD_MPIIO_IND, comm_);

        // set step to last step in file if negative
        if (step < 0) { step = H5PartGetNumSteps(h5File_) - 1; }

        H5PartSetStep(h5File_, step);

        globalCount_ = H5PartGetNumParticles(h5File_);
        if (globalCount_ < 1) { throw std::runtime_error("no particles in input file found\n"); }

        MPI_Comm_rank(comm_, &rank_);
        MPI_Comm_size(comm_, &numRanks_);

        std::tie(firstIndex_, lastIndex_) = partitionRange(globalCount_, rank_, numRanks_);
        localCount_                       = lastIndex_ - firstIndex_;
        std::cout << "Rank: " << rank_ << ", lastIndex_: " << lastIndex_ << ", firstIndex_: " << firstIndex_
                  << std::endl;

        h5z_              = fileutils::openHDF5File(path, comm_);
        h5z_.step         = step;
        h5z_.start        = firstIndex_;
        h5z_.numParticles = localCount_;

        H5PartSetView(h5File_, firstIndex_, lastIndex_ - 1);
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
        auto           attributes = fileutils::fileAttributeNames(h5File_);
        size_t         attrIndex  = std::find(attributes.begin(), attributes.end(), key) - attributes.begin();
        h5part_int64_t typeId, attrSize;
        char           dummy[256];
        H5PartGetFileAttribInfo(h5File_, attrIndex, dummy, 256, &typeId, &attrSize);
        return attrSize;
    }

    int64_t stepAttributeSize(const std::string& key) override
    {
        auto   attributes = fileutils::stepAttributeNames(h5File_);
        size_t attrIndex  = std::find(attributes.begin(), attributes.end(), key) - attributes.begin();
        if (attrIndex == attributes.size()) { throw std::out_of_range("Attribute " + key + " does not exist\n"); }

        h5part_int64_t typeId, attrSize;
        char           dummy[256];
        H5PartGetStepAttribInfo(h5File_, attrIndex, dummy, 256, &typeId, &attrSize);
        return attrSize;
    }

    void fileAttribute(const std::string& key, FieldType val, int64_t size) override
    {
        if (size != fileAttributeSize(key)) { throw std::runtime_error("File attribute size is inconsistent: " + key); }
        auto err = std::visit([this, &key](auto arg) { return H5PartReadFileAttrib(h5File_, key.c_str(), arg); }, val);
        if (err != H5PART_SUCCESS) { throw std::out_of_range("Could not read file attribute: " + key); }
    }

    void stepAttribute(const std::string& key, FieldType val, int64_t size) override
    {
        if (size != stepAttributeSize(key)) { throw std::runtime_error("step attribute size is inconsistent: " + key); }
        auto err = std::visit(
            [this, size, &key](auto arg)
            {
                int64_t memTypeId  = fileutils::H5PartType<std::decay_t<decltype(*arg)>>{};
                int64_t fileTypeId = stepAttributeType(key);
                if (memTypeId != fileTypeId)
                {
                    if (memTypeId == fileutils::H5PartType<float>{} && fileTypeId == fileutils::H5PartType<double>{})
                    {
                        double tmp[size];
                        auto   err = H5PartReadStepAttrib(h5File_, key.c_str(), tmp);
                        std::copy_n(tmp, size, arg);
                        return err;
                    }
                    else if (memTypeId == fileutils::H5PartType<double>{} &&
                             fileTypeId == fileutils::H5PartType<float>{})
                    {
                        float tmp[size];
                        auto  err = H5PartReadStepAttrib(h5File_, key.c_str(), tmp);
                        std::copy_n(tmp, size, arg);
                        return err;
                    }
                    else
                    {
                        throw std::runtime_error("attribute type of " + key + " in file is " +
                                                 fileutils::H5PartTypeToString(fileTypeId) + ", but should be " +
                                                 fileutils::H5PartTypeToString(memTypeId) + "\n");
                    }
                }

                return H5PartReadStepAttrib(h5File_, key.c_str(), arg);
            },
            val);
        if (err != H5PART_SUCCESS) { throw std::runtime_error("Could not read step attribute: " + key); }
    }

    void readField(const std::string& key, FieldType field) override
    {
        auto err = std::visit([this, &key](auto arg) { return fileutils::readHDF5Field(h5z_, key, arg, globalCount_); },
                              field);
        if (err != H5PART_SUCCESS) { throw std::runtime_error("Could not read field: " + key); }
    }

    uint64_t localNumParticles() override { return localCount_; }

    uint64_t globalNumParticles() override { return globalCount_; }

    void closeStep() override
    {
        if (rank_ == 0)
        {
            std::cout << "File init elapse: " << fileInitTime_ << ", writing elapse: " << writeTime_ << std::endl;
        }
        fileutils::closeHDF5File(h5z_, true);
    }

private:
    h5part_int64_t stepAttributeType(const std::string& key)
    {
        auto           attributes = fileutils::stepAttributeNames(h5File_);
        size_t         attrIndex  = std::find(attributes.begin(), attributes.end(), key) - attributes.begin();
        h5part_int64_t typeId, attrSize;
        char           dummy[256];
        H5PartGetStepAttribInfo(h5File_, attrIndex, dummy, 256, &typeId, &attrSize);
        return typeId;
    }

    MPI_Comm comm_;

    uint64_t    firstIndex_, lastIndex_;
    uint64_t    localCount_;
    uint64_t    globalCount_;
    int         rank_, numRanks_;
    std::string pathStep_;
    double      fileInitTime_, writeTime_;

    H5PartFile*        h5File_;
    fileutils::H5ZType h5z_;
};

} // namespace sphexa
