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

class H5PartWriter : public IFileWriter
{
public:
    using Base      = IFileWriter;
    using FieldType = typename Base::FieldType;

    H5PartWriter(MPI_Comm comm)
        : comm_(comm)
    {
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

        if (std::filesystem::exists(path))
        {
            h5File_ = fileutils::openH5Part(path, H5PART_APPEND | H5PART_VFD_MPIIO_IND, comm_);
        }
        else { h5File_ = fileutils::openH5Part(path, H5PART_WRITE | H5PART_VFD_MPIIO_IND, comm_); }

        // create the next step
        h5part_int64_t numSteps = H5PartGetNumSteps(h5File_);
        H5PartSetStep(h5File_, numSteps);

        uint64_t numParticles = lastIndex - firstIndex;
        // set number of particles that each rank will write
        H5PartSetNumParticles(h5File_, numParticles);
    }

    void stepAttribute(const std::string& key, FieldType val, int64_t size) override
    {
        std::visit([this, &key, size](auto arg) { fileutils::sphexaWriteStepAttrib(h5File_, key.c_str(), arg, size); },
                   val);
    }

    void writeField(const std::string& key, FieldType field, int = 0) override
    {
        std::visit([this, &key](auto arg) { fileutils::writeH5PartField(h5File_, key, arg + firstIndex_); }, field);
    }

    void closeStep() override { H5PartCloseFile(h5File_); }

private:
    MPI_Comm comm_;

    size_t      firstIndex_, lastIndex_;
    std::string pathStep_;

    H5PartFile* h5File_;
};

class H5PartReader : public IFileReader
{
public:
    using Base      = IFileReader;
    using FieldType = typename Base::FieldType;

    H5PartReader(MPI_Comm comm)
        : comm_(comm)
    {
    }

    void setStep(std::string path, int step) override
    {
        pathStep_ = path;
        h5File_   = fileutils::openH5Part(path, H5PART_READ | H5PART_VFD_MPIIO_IND, comm_);

        // set step to last step in file if negative
        if (step < 0) { step = H5PartGetNumSteps(h5File_) - 1; }
        H5PartSetStep(h5File_, step);

        globalCount_ = H5PartGetNumParticles(h5File_);
        if (globalCount_ < 1) { throw std::runtime_error("no particles in input file found\n"); }

        int rank, numRanks;
        MPI_Comm_rank(comm_, &rank);
        MPI_Comm_size(comm_, &numRanks);

        std::tie(firstIndex_, lastIndex_) = partitionRange(globalCount_, rank, numRanks);
        localCount_                       = lastIndex_ - firstIndex_;

        H5PartSetView(h5File_, firstIndex_, lastIndex_ - 1);
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
        auto           attributes = fileutils::stepAttributeNames(h5File_);
        size_t         attrIndex  = std::find(attributes.begin(), attributes.end(), key) - attributes.begin();
        h5part_int64_t typeId, attrSize;
        char           dummy[256];
        H5PartGetStepAttribInfo(h5File_, attrIndex, dummy, 256, &typeId, &attrSize);
        return attrSize;
    }

    void fileAttribute(const std::string& key, FieldType val, int64_t size) override
    {
        if (size != fileAttributeSize(key)) { throw std::runtime_error("File attribute size is inconsistent: " + key); }
        auto err = std::visit([this, &key](auto arg) { return H5PartReadFileAttrib(h5File_, key.c_str(), arg); }, val);
        if (err != H5PART_SUCCESS) { throw std::runtime_error("Could not read file attribute: " + key); }
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
        auto err = std::visit([this, &key](auto arg) { return fileutils::readH5PartField(h5File_, key, arg); }, field);
        if (err != H5PART_SUCCESS) { throw std::runtime_error("Could not read field: " + key); }
    }

    uint64_t localNumParticles() override { return localCount_; }

    uint64_t globalNumParticles() override { return globalCount_; }

    void closeStep() override { H5PartCloseFile(h5File_); }

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
    std::string pathStep_;

    H5PartFile* h5File_;
};

} // namespace sphexa
