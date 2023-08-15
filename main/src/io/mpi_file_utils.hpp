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
 * @brief parallel file I/O utility functions based on H5Part
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <vector>
#include <iostream>

#include "H5Part.h"
#include "H5Cpp.h"
#include "H5Zzfp_plugin.h"

namespace sphexa
{
namespace fileutils
{

template<class T>
struct H5PartType
{
};

struct H5ZType {
    hid_t file_id;
    hid_t dset_id;
    hid_t dspace_id;
    hid_t group_id;
    hid_t cpid;
    hid_t status;
};



static hid_t setup_filter(int n, hsize_t *chunk, int zfpmode,
    double rate, double acc, unsigned int prec,
    unsigned int minbits, unsigned int maxbits, unsigned int maxprec, int minexp)
{
    hid_t cpid;
    hid_t status;

    /* setup dataset creation properties */
    cpid = H5Pcreate(H5P_DATASET_CREATE);
    status = H5Pset_chunk(cpid, n, chunk);


    unsigned int cd_values[10];
    size_t cd_nelmts = 10;

    /* setup zfp filter via generic (cd_values) interface */
    if (zfpmode == H5Z_ZFP_MODE_RATE)
        H5Pset_zfp_rate_cdata(rate, cd_nelmts, cd_values);
    else if (zfpmode == H5Z_ZFP_MODE_PRECISION)
        H5Pset_zfp_precision_cdata(prec, cd_nelmts, cd_values);
    else if (zfpmode == H5Z_ZFP_MODE_ACCURACY)
        H5Pset_zfp_accuracy_cdata(acc, cd_nelmts, cd_values);
    else if (zfpmode == H5Z_ZFP_MODE_EXPERT)
        H5Pset_zfp_expert_cdata(minbits, maxbits, maxprec, minexp, cd_nelmts, cd_values);
    else if (zfpmode == H5Z_ZFP_MODE_REVERSIBLE)
        H5Pset_zfp_reversible_cdata(cd_nelmts, cd_values);
    else
        cd_nelmts = 0; /* causes default behavior of ZFP library */

    /* print cd-values array used for filter */
    printf("\n%d cd_values=", (int) cd_nelmts);
    for (int i = 0; i < (int) cd_nelmts; i++)
        printf("%u,", cd_values[i]);
    printf("\n");
    const char* env_p = std::getenv("HDF5_PLUGIN_PATH");
    std::cout << "Your PATH is: " << env_p << '\n';

    /* Add filter to the pipeline via generic interface */
    // status = H5Pset_filter(cpid, H5Z_FILTER_ZFP, H5Z_FLAG_MANDATORY, cd_nelmts, cd_values);
        
    return cpid;
}

static H5ZType create_h5z_file(std::string fileName, std::string groupName, std::string datasetName, int nRow, int nCol)
{
    // Setup filter
    hsize_t chunk = nRow;
    int zfpmode = H5Z_ZFP_MODE_RATE;
    double rate = 4;
    double acc = 0;
    unsigned int prec = 11;
    unsigned int minbits = 0;
    unsigned int maxbits = 4171;
    unsigned int maxprec = 64;
    int minexp = -1074;

    H5ZType h5z;

    h5z.cpid = setup_filter(1, &chunk, zfpmode, rate, acc, prec, minbits, maxbits, maxprec, minexp);

    herr_t status;

    // Open a file
    h5z.file_id = H5Fcreate(fileName.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    // Create a group
    h5z.group_id = H5Gcreate2(h5z.file_id, groupName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    return h5z;
}

static H5ZType open_h5z_file(std::string fileName, std::string groupName, std::string datasetName, int nRow, int nCol)
{

    H5ZType h5z;
    herr_t status;

    // Setup dataset dimensions and input data
    int ndims = nCol;
    hsize_t dims[ndims];
    for (size_t i=0; i<ndims; i++) {
        dims[i] = nRow;
    }
    std::vector<double> data(nRow, nCol);

    // Open a file
    h5z.file_id = H5Fopen(fileName.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);

    // Create a group
    h5z.group_id = H5Gcreate2(h5z.file_id, groupName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    return h5z;
}

static void add_filter(H5ZType& h5z) {
    // Setup filter
    hsize_t chunk_dims[2] ={2, 5};
    int zfpmode = H5Z_ZFP_MODE_RATE;
    double rate = 4;
    double acc = 0;
    unsigned int prec = 11;
    unsigned int minbits = 0;
    unsigned int maxbits = 4171;
    unsigned int maxprec = 64;
    int minexp = -1074;

    h5z.cpid = setup_filter(1, chunk_dims, zfpmode, rate, acc, prec, minbits, maxbits, maxprec, minexp);
}


// nRow: num of particles. nCol: num of data fields.
// Setup dataset dimensions
static void add_h5z_step(H5ZType& h5z, std::string fieldName, hsize_t nRow, hsize_t nCol = 0) {
    hsize_t dims[2] = {nRow, nCol};
    hsize_t maxdims[2] = {H5S_UNLIMITED, H5S_UNLIMITED};
    h5z.dspace_id = H5Screate_simple(1, dims, maxdims);

    // dataset has to be chunked to be extendable
    hsize_t chunk_dims[2] ={2, 5};

    // // Modify dataset creation properties, i.e. enable chunking
    // hid_t prop   = H5Pcreate(H5P_DATASET_CREATE);
    // h5z.status = H5Pset_chunk(prop, 1, chunk_dims);

    add_filter(h5z);

    h5z.dset_id = H5Dcreate2(
        h5z.group_id, fieldName.c_str(), H5T_STD_I32BE, h5z.dspace_id, H5P_DEFAULT, h5z.cpid, H5P_DEFAULT);
}

static void write_h5z_field(H5ZType& h5z, const float* field) {
    h5z.status = H5Dwrite(h5z.dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, field);
}

static void write_h5z_field(H5ZType& h5z, const double* field) {
    hsize_t      size[2];
    hsize_t      old_size[2];
    hsize_t      max_size[2];
    H5Sget_simple_extent_dims(h5z.dspace_id, old_size, max_size);
    size[0]   = old_size[0];
    size[1]   = old_size[1] + 1;

    H5Dset_extent(h5z.dset_id, size);

    h5z.status = H5Dwrite(h5z.dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, field);
}

static void write_h5z_field(H5ZType& h5z, const char* field) {
    h5z.status = H5Dwrite(h5z.dset_id, H5T_NATIVE_CHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, field);
}

static void write_h5z_field(H5ZType& h5z, const int* field) {
    h5z.status = H5Dwrite(h5z.dset_id, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, field);
}

static void write_h5z_field(H5ZType& h5z, const unsigned int* field) {
    h5z.status = H5Dwrite(h5z.dset_id, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, field);
}

static void write_h5z_field(H5ZType& h5z, const long int* field) {
    h5z.status = H5Dwrite(h5z.dset_id, H5T_NATIVE_LONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, field);
}

static void write_h5z_field(H5ZType& h5z, const long unsigned int* field) {
    h5z.status = H5Dwrite(h5z.dset_id, H5T_NATIVE_ULONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, field);
}

static void write_h5z_attrib(H5ZType& h5z, const std::string& fieldName, const hsize_t attrib_nelem, const float* field) {
    hid_t space_id = H5Screate_simple ( 1, &attrib_nelem, NULL );
    hid_t attrib_properties_id = H5Pcreate(H5P_ATTRIBUTE_CREATE);
    hid_t attrib_id = H5Acreate(
        h5z.file_id,
        fieldName.c_str(),
        H5T_NATIVE_FLOAT,
        space_id,
        attrib_properties_id,
        H5P_DEFAULT
    );
    H5Aclose(attrib_id);
    H5Sclose(space_id);
    H5Pclose(attrib_properties_id);
}

static void write_h5z_attrib(H5ZType& h5z, const std::string& fieldName, const hsize_t attrib_nelem, const double* field) {
    hid_t space_id = H5Screate_simple ( 1, &attrib_nelem, NULL );
    hid_t attrib_properties_id = H5Pcreate(H5P_ATTRIBUTE_CREATE);
    hid_t attrib_id = H5Acreate(
        h5z.file_id,
        fieldName.c_str(),
        H5T_NATIVE_DOUBLE,
        space_id,
        attrib_properties_id,
        H5P_DEFAULT
    );
    H5Aclose(attrib_id);
    H5Sclose(space_id);
    H5Pclose(attrib_properties_id);
}

static void write_h5z_attrib(H5ZType& h5z, const std::string& fieldName, const hsize_t attrib_nelem, const char* field) {
    hid_t space_id = H5Screate_simple ( 1, &attrib_nelem, NULL );
    hid_t attrib_properties_id = H5Pcreate(H5P_ATTRIBUTE_CREATE);
    hid_t attrib_id = H5Acreate(
        h5z.file_id,
        fieldName.c_str(),
        H5T_NATIVE_CHAR,
        space_id,
        attrib_properties_id,
        H5P_DEFAULT
    );
    H5Aclose(attrib_id);
    H5Sclose(space_id);
    H5Pclose(attrib_properties_id);
}

static void write_h5z_attrib(H5ZType& h5z, const std::string& fieldName, const hsize_t attrib_nelem, const int* field) {
    hid_t space_id = H5Screate_simple ( 1, &attrib_nelem, NULL );
    hid_t attrib_properties_id = H5Pcreate(H5P_ATTRIBUTE_CREATE);
    hid_t attrib_id = H5Acreate(
        h5z.file_id,
        fieldName.c_str(),
        H5T_NATIVE_INT32,
        space_id,
        attrib_properties_id,
        H5P_DEFAULT
    );
    H5Aclose(attrib_id);
    H5Sclose(space_id);
    H5Pclose(attrib_properties_id);
}
static void write_h5z_attrib(H5ZType& h5z, const std::string& fieldName, const hsize_t attrib_nelem, const unsigned int* field) {
    hid_t space_id = H5Screate_simple ( 1, &attrib_nelem, NULL );
    hid_t attrib_properties_id = H5Pcreate(H5P_ATTRIBUTE_CREATE);
    hid_t attrib_id = H5Acreate(
        h5z.file_id,
        fieldName.c_str(),
        H5T_NATIVE_UINT32,
        space_id,
        attrib_properties_id,
        H5P_DEFAULT
    );
    H5Aclose(attrib_id);
    H5Sclose(space_id);
    H5Pclose(attrib_properties_id);
}

static void write_h5z_attrib(H5ZType& h5z, const std::string& fieldName, const hsize_t attrib_nelem, const long int* field) {
    hid_t space_id = H5Screate_simple ( 1, &attrib_nelem, NULL );
    hid_t attrib_properties_id = H5Pcreate(H5P_ATTRIBUTE_CREATE);
    hid_t attrib_id = H5Acreate(
        h5z.file_id,
        fieldName.c_str(),
        H5T_NATIVE_LONG,
        space_id,
        attrib_properties_id,
        H5P_DEFAULT
    );
    H5Aclose(attrib_id);
    H5Sclose(space_id);
    H5Pclose(attrib_properties_id);
}

static void write_h5z_attrib(H5ZType& h5z, const std::string& fieldName, const hsize_t attrib_nelem, const long unsigned int* field) {
    hid_t space_id = H5Screate_simple ( 1, &attrib_nelem, NULL );
    hid_t attrib_properties_id = H5Pcreate(H5P_ATTRIBUTE_CREATE);
    hid_t attrib_id = H5Acreate(
        h5z.file_id,
        fieldName.c_str(),
        H5T_NATIVE_ULONG,
        space_id,
        attrib_properties_id,
        H5P_DEFAULT
    );
    H5Aclose(attrib_id);
    H5Sclose(space_id);
    H5Pclose(attrib_properties_id);
}

static void close_zfp(H5ZType& h5z) {
    // Close the dataset and group
    h5z.status = H5Sclose(h5z.dspace_id);
    h5z.status = H5Dclose(h5z.dset_id);
    h5z.status = H5Gclose(h5z.group_id);
    h5z.status = H5Pclose(h5z.cpid);

    // Close the file
    h5z.status = H5Fclose(h5z.file_id);
}

std::string H5PartTypeToString(h5part_int64_t type)
{
    if (type == H5PART_FLOAT64) { return "C++: double / python: np.float64"; }
    if (type == H5PART_FLOAT32) { return "C++: float / python: np.float32"; }
    if (type == H5PART_INT32) { return "C++: int / python: np.int32"; }
    if (type == H5PART_INT64) { return "C++: int64_t / python: np.int64"; }
    if (type == H5PART_CHAR) { return "C++: char / python: np.int8"; }

    return "H5PART_UNKNOWN";
}

template<>
struct H5PartType<double>
{
    operator decltype(H5PART_FLOAT64)() const noexcept { return H5PART_FLOAT64; }
};

template<>
struct H5PartType<float>
{
    operator decltype(H5PART_FLOAT32)() const noexcept { return H5PART_FLOAT32; }
};

template<>
struct H5PartType<char>
{
    operator decltype(H5PART_CHAR)() const noexcept { return H5PART_CHAR; }
};

template<>
struct H5PartType<int>
{
    operator decltype(H5PART_INT32)() const noexcept { return H5PART_INT32; }
};

template<>
struct H5PartType<unsigned>
{
    operator decltype(H5PART_INT32)() const noexcept { return H5PART_INT32; }
};

template<>
struct H5PartType<int64_t>
{
    operator decltype(H5PART_INT64)() const noexcept { return H5PART_INT64; }
};

template<>
struct H5PartType<uint64_t>
{
    operator decltype(H5PART_INT64)() const noexcept { return H5PART_INT64; }
};

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

//! @brief return the names of all file attributes in @p h5_file
std::vector<std::string> fileAttributeNames(H5PartFile* h5_file)
{
    auto numAttributes = H5PartGetNumFileAttribs(h5_file);

    std::vector<std::string> setNames(numAttributes);
    for (size_t fi = 0; fi < numAttributes; ++fi)
    {
        int            maxlen = 256;
        char           attrName[maxlen];
        h5part_int64_t typeId, attrSize;

        H5PartGetFileAttribInfo(h5_file, fi, attrName, maxlen, &typeId, &attrSize);
        setNames[fi] = std::string(attrName);
    }

    return setNames;
}

//! @brief return the names of all file attributes in @p h5_file
std::vector<std::string> stepAttributeNames(H5PartFile* h5_file)
{
    auto numAttributes = H5PartGetNumStepAttribs(h5_file);

    std::vector<std::string> setNames(numAttributes);
    for (size_t fi = 0; fi < numAttributes; ++fi)
    {
        int            maxlen = 256;
        char           attrName[maxlen];
        h5part_int64_t typeId, attrSize;

        H5PartGetStepAttribInfo(h5_file, fi, attrName, maxlen, &typeId, &attrSize);
        setNames[fi] = std::string(attrName);
    }

    return setNames;
}

/* read fields */

inline h5part_int64_t readH5PartField(H5PartFile* h5_file, const std::string& fieldName, double* field)
{
    static_assert(std::is_same_v<double, h5part_float64_t>);
    return H5PartReadDataFloat64(h5_file, fieldName.c_str(), field);
}

inline h5part_int64_t readH5PartField(H5PartFile* h5_file, const std::string& fieldName, float* field)
{
    static_assert(std::is_same_v<float, h5part_float32_t>);
    return H5PartReadDataFloat32(h5_file, fieldName.c_str(), field);
}

inline h5part_int64_t readH5PartField(H5PartFile* h5_file, const std::string& fieldName, char* /*field*/)
{
    throw std::runtime_error("H5Part read char field not implemented");
}

inline h5part_int64_t readH5PartField(H5PartFile* h5_file, const std::string& fieldName, int* field)
{
    static_assert(std::is_same_v<int, h5part_int32_t>);
    return H5PartReadDataInt32(h5_file, fieldName.c_str(), field);
}

inline h5part_int64_t readH5PartField(H5PartFile* h5_file, const std::string& fieldName, int64_t* field)
{
    return H5PartReadDataInt64(h5_file, fieldName.c_str(), (h5part_int64_t*)field);
}

inline h5part_int64_t readH5PartField(H5PartFile* h5_file, const std::string& fieldName, unsigned* field)
{
    return H5PartReadDataInt32(h5_file, fieldName.c_str(), (h5part_int32_t*)field);
}

inline h5part_int64_t readH5PartField(H5PartFile* h5_file, const std::string& fieldName, uint64_t* field)
{
    return H5PartReadDataInt64(h5_file, fieldName.c_str(), (h5part_int64_t*)field);
}

/* write fields */

inline h5part_int64_t writeH5PartField(H5PartFile* h5_file, const std::string& fieldName, const double* field)
{
    static_assert(std::is_same_v<double, h5part_float64_t>);
    return H5PartWriteDataFloat64(h5_file, fieldName.c_str(), field);
}

inline h5part_int64_t writeH5PartField(H5PartFile* h5_file, const std::string& fieldName, const float* field)
{
    static_assert(std::is_same_v<float, h5part_float32_t>);
    return H5PartWriteDataFloat32(h5_file, fieldName.c_str(), field);
}

inline h5part_int64_t writeH5PartField(H5PartFile* h5_file, const std::string& fieldName, const char* /*field*/)
{
    throw std::runtime_error("H5Part write char field not implemented");
}

inline h5part_int64_t writeH5PartField(H5PartFile* h5_file, const std::string& fieldName, const int* field)
{
    static_assert(std::is_same_v<int, h5part_int32_t>);
    return H5PartWriteDataInt32(h5_file, fieldName.c_str(), field);
}

inline h5part_int64_t writeH5PartField(H5PartFile* h5_file, const std::string& fieldName, const int64_t* field)
{
    return H5PartWriteDataInt64(h5_file, fieldName.c_str(), (const h5part_int64_t*)field);
}

inline h5part_int64_t writeH5PartField(H5PartFile* h5_file, const std::string& fieldName, const unsigned* field)
{
    return H5PartWriteDataInt32(h5_file, fieldName.c_str(), (const h5part_int32_t*)field);
}

inline h5part_int64_t writeH5PartField(H5PartFile* h5_file, const std::string& fieldName, const uint64_t* field)
{
    return H5PartWriteDataInt64(h5_file, fieldName.c_str(), (const h5part_int64_t*)field);
}

/* write step attributes */

auto sphexaWriteStepAttrib(H5PartFile* h5_file, const std::string& name, const double* value, size_t numElements)
{
    return H5PartWriteStepAttrib(h5_file, name.c_str(), H5PART_FLOAT64, value, numElements);
}

auto sphexaWriteStepAttrib(H5PartFile* h5_file, const std::string& name, const float* value, size_t numElements)
{
    return H5PartWriteStepAttrib(h5_file, name.c_str(), H5PART_FLOAT32, value, numElements);
}

auto sphexaWriteStepAttrib(H5PartFile* h5_file, const std::string& name, const char* value, size_t numElements)
{
    return H5PartWriteStepAttrib(h5_file, name.c_str(), H5PART_CHAR, value, numElements);
}

auto sphexaWriteStepAttrib(H5PartFile* h5_file, const std::string& name, const int* value, size_t numElements)
{
    return H5PartWriteStepAttrib(h5_file, name.c_str(), H5PART_INT32, value, numElements);
}

auto sphexaWriteStepAttrib(H5PartFile* h5_file, const std::string& name, const int64_t* value, size_t numElements)
{
    return H5PartWriteStepAttrib(h5_file, name.c_str(), H5PART_INT64, value, numElements);
}

auto sphexaWriteStepAttrib(H5PartFile* h5_file, const std::string& name, const unsigned* value, size_t numElements)
{
    return H5PartWriteStepAttrib(h5_file, name.c_str(), H5PART_INT32, value, numElements);
}

auto sphexaWriteStepAttrib(H5PartFile* h5_file, const std::string& name, const uint64_t* value, size_t numElements)
{
    return H5PartWriteStepAttrib(h5_file, name.c_str(), H5PART_INT64, value, numElements);
}

/* write file attributes */

auto sphexaWriteFileAttrib(H5PartFile* h5_file, const std::string& name, const double* value, size_t numElements)
{
    return H5PartWriteFileAttrib(h5_file, name.c_str(), H5PART_FLOAT64, value, numElements);
}

auto sphexaWriteFileAttrib(H5PartFile* h5_file, const std::string& name, const float* value, size_t numElements)
{
    return H5PartWriteFileAttrib(h5_file, name.c_str(), H5PART_FLOAT32, value, numElements);
}

auto sphexaWriteFileAttrib(H5PartFile* h5_file, const std::string& name, const char* value, size_t numElements)
{
    return H5PartWriteFileAttrib(h5_file, name.c_str(), H5PART_CHAR, value, numElements);
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
