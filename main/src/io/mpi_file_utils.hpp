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
 * @brief parallel file I/O utility functions based on H5Part
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <filesystem>
#include <variant>
#include "H5Part.h"

namespace sphexa
{
namespace fileutils
{

//! @brief return the names of all datasets in @p h5_file
std::vector<std::string> datasetNames(H5PartFile* h5_file)
{
    auto numSets = H5PartGetNumDatasets(h5_file);

    std::vector<std::string> setNames(numSets);
    for (size_t fi = 0; fi < numSets; ++fi)
    {
        int  maxlen = 256;
        char fieldName[maxlen];
        H5PartGetDatasetName(h5_file, fi, fieldName, maxlen);
        setNames[fi] = std::string(fieldName);
    }

    return setNames;
}

//! @brief return the names of all datasets in @p h5_file
std::vector<std::string> fileAttributeNames(H5PartFile* h5_file)
{
    auto numAttributes = H5PartGetNumFileAttribs(h5_file);

    std::vector<std::string> setNames(numAttributes);
    for (size_t fi = 0; fi < numAttributes; ++fi)
    {
        int            maxlen = 256;
        char           attrName[maxlen];
        h5part_int64_t typeId, attrSize;

        H5PartGetFileAttribInfo(h5_file, fi, attrName, 256, &typeId, &attrSize);
        setNames[fi] = std::string(attrName);
    }

    return setNames;
}

inline h5part_int64_t readH5PartField(H5PartFile* h5_file, const std::string& fieldName, int* field)
{
    static_assert(std::is_same_v<int, h5part_int32_t>);
    return H5PartReadDataInt32(h5_file, fieldName.c_str(), field);
}

inline h5part_int64_t readH5PartField(H5PartFile* h5_file, const std::string& fieldName, float* field)
{
    static_assert(std::is_same_v<float, h5part_float32_t>);
    return H5PartReadDataFloat32(h5_file, fieldName.c_str(), field);
}

inline h5part_int64_t readH5PartField(H5PartFile* h5_file, const std::string& fieldName, double* field)
{
    static_assert(std::is_same_v<double, h5part_float64_t>);
    return H5PartReadDataFloat64(h5_file, fieldName.c_str(), field);
}

inline h5part_int64_t writeH5PartField(H5PartFile* h5_file, const std::string& fieldName, const int* field)
{
    static_assert(std::is_same_v<int, h5part_int32_t>);
    return H5PartWriteDataInt32(h5_file, fieldName.c_str(), field);
}

inline h5part_int64_t writeH5PartField(H5PartFile* h5_file, const std::string& fieldName, const unsigned* field)
{
    return H5PartWriteDataInt32(h5_file, fieldName.c_str(), (const h5part_int32_t*)field);
}

inline h5part_int64_t writeH5PartField(H5PartFile* h5_file, const std::string& fieldName, const uint64_t* field)
{
    return H5PartWriteDataInt64(h5_file, fieldName.c_str(), (const h5part_int64_t*)field);
}

inline h5part_int64_t writeH5PartField(H5PartFile* h5_file, const std::string& fieldName, const float* field)
{
    static_assert(std::is_same_v<float, h5part_float32_t>);
    return H5PartWriteDataFloat32(h5_file, fieldName.c_str(), field);
}

inline h5part_int64_t writeH5PartField(H5PartFile* h5_file, const std::string& fieldName, const double* field)
{
    static_assert(std::is_same_v<double, h5part_float64_t>);
    return H5PartWriteDataFloat64(h5_file, fieldName.c_str(), field);
}

void sphexaWriteStepAttrib(H5PartFile* h5_file, const std::string& name, double* value, size_t numElements)
{
    H5PartWriteStepAttrib(h5_file, name.c_str(), H5PART_FLOAT64, value, numElements);
}

void sphexaWriteStepAttrib(H5PartFile* h5_file, const std::string& name, float* value, size_t numElements)
{
    H5PartWriteStepAttrib(h5_file, name.c_str(), H5PART_FLOAT32, value, numElements);
}

void sphexaWriteFileAttrib(H5PartFile* h5_file, const std::string& name, const double* value, size_t numElements)
{
    H5PartWriteFileAttrib(h5_file, name.c_str(), H5PART_FLOAT64, value, numElements);
}

void sphexaWriteFileAttrib(H5PartFile* h5_file, const std::string& name, const float* value, size_t numElements)
{
    H5PartWriteFileAttrib(h5_file, name.c_str(), H5PART_FLOAT32, value, numElements);
}

void sphexaWriteFileAttrib(H5PartFile* h5_file, const std::string& name, const char* value, size_t numElements)
{
    H5PartWriteFileAttrib(h5_file, name.c_str(), H5PART_CHAR, value, numElements);
}

//! @brief Open in parallel mode if supported, otherwise serial if numRanks == 1
H5PartFile* openH5Part(const std::string& path, h5part_int64_t mode, MPI_Comm comm)
{
    const char* h5_fname = path.c_str();
    H5PartFile* h5_file  = nullptr;

#ifdef H5PART_PARALLEL_IO
    h5_file = H5PartOpenFileParallel(h5_fname, mode, comm);
#else
    int numRanks;
    MPI_Comm_size(comm, &numRanks);
    if (numRanks > 1)
    {
        throw std::runtime_error("Cannot open HDF5 file on multiple ranks without parallel HDF5 support\n");
    }
    h5_file = H5PartOpenFile(h5_fname, mode);
#endif

    return h5_file;
}

template<class Dataset, class T>
void writeH5Part(Dataset& d, size_t firstIndex, size_t lastIndex, const cstone::Box<T>& box, const std::string& path)
{
    H5PartFile* h5_file = nullptr;

    if (std::filesystem::exists(path)) { h5_file = openH5Part(path, H5PART_APPEND, d.comm); }
    else { h5_file = openH5Part(path, H5PART_WRITE, d.comm); }

    // create the next step
    h5part_int64_t numSteps = H5PartGetNumSteps(h5_file);
    H5PartSetStep(h5_file, numSteps);

    sphexaWriteStepAttrib(h5_file, "time", &d.ttot, 1);
    sphexaWriteStepAttrib(h5_file, "minDt", &d.minDt, 1);
    sphexaWriteStepAttrib(h5_file, "minDt_m1", &d.minDt_m1, 1);
    sphexaWriteStepAttrib(h5_file, "gravConstant", &d.g, 1);
    sphexaWriteStepAttrib(h5_file, "gamma", &d.gamma, 1);
    // record the actual SPH-iteration as step attribute
    H5PartWriteStepAttrib(h5_file, "step", H5PART_INT64, &d.iteration, 1);

    // record the global coordinate bounding box
    double extents[6] = {box.xmin(), box.xmax(), box.ymin(), box.ymax(), box.zmin(), box.zmax()};
    H5PartWriteStepAttrib(h5_file, "box", H5PART_FLOAT64, extents, 6);
    h5part_int32_t boundaries[3] = {static_cast<h5part_int32_t>(box.boundaryX()),
                                    static_cast<h5part_int32_t>(box.boundaryY()),
                                    static_cast<h5part_int32_t>(box.boundaryZ())};
    H5PartWriteStepAttrib(h5_file, "boundaryType", H5PART_INT32, boundaries, 3);

    const h5part_int64_t h5_num_particles = lastIndex - firstIndex;
    // set number of particles that each rank will write
    H5PartSetNumParticles(h5_file, h5_num_particles);

    auto fieldPointers = getOutputArrays(d);
    for (size_t fidx = 0; fidx < fieldPointers.size(); ++fidx)
    {
        const std::string& fieldName = d.outputFieldNames[fidx];
        std::visit([&h5_file, &fieldName, firstIndex](auto& arg)
                   { writeH5PartField(h5_file, fieldName, arg + firstIndex); },
                   fieldPointers[fidx]);
    }

    H5PartCloseFile(h5_file);
}

//! @brief read x,y,z coordinates from an H5Part file (at step 0)
template<class Vector>
void readTemplateBlock(std::string block, Vector& x, Vector& y, Vector& z)
{
    H5PartFile* h5_file = nullptr;
    h5_file             = H5PartOpenFile(block.c_str(), H5PART_READ);
    H5PartSetStep(h5_file, 0);
    size_t blockSize = H5PartGetNumParticles(h5_file);
    x.resize(blockSize);
    y.resize(blockSize);
    z.resize(blockSize);

    // read the template block
    fileutils::readH5PartField(h5_file, "x", x.data());
    fileutils::readH5PartField(h5_file, "y", y.data());
    fileutils::readH5PartField(h5_file, "z", z.data());
    H5PartCloseFile(h5_file);
}

} // namespace fileutils
} // namespace sphexa
