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

#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>
#include <mpi.h>
#include "H5Part.h"
#include "hdf5.h"
#include "H5Zzfp.h"

namespace sphexa
{
namespace fileutils
{

template<class T>
struct H5PartType
{
};

enum class CompressionMethod
{
    none,
    deflate,
    gzip,
    szip,
    zfp
};

struct H5ZType {
    CompressionMethod compression;
    int compressionParam;
    hid_t file_id;
    hid_t group_id;
    hid_t dset_id;
    hid_t dspace_id;
    hid_t cpid;
    hid_t status;
    size_t numParticles;
    size_t step;
};

static void setupZFP(H5ZType& h5z, int zfpmode,
    double rate, double acc, unsigned int prec,
    unsigned int minbits, unsigned int maxbits, unsigned int maxprec, int minexp)
{
    hid_t status;
    unsigned int cd_values[10];
    size_t cd_nelmts = 10;

    H5Z_zfp_initialize();

    /* Setup the filter using properties interface. These calls also add
       the filter to the pipeline */
    if (zfpmode == H5Z_ZFP_MODE_RATE)
        H5Pset_zfp_rate(h5z.cpid, rate);
    else if (zfpmode == H5Z_ZFP_MODE_PRECISION)
        H5Pset_zfp_precision(h5z.cpid, prec);
    else if (zfpmode == H5Z_ZFP_MODE_ACCURACY)
        H5Pset_zfp_accuracy(h5z.cpid, acc);
    else if (zfpmode == H5Z_ZFP_MODE_EXPERT)
        H5Pset_zfp_expert(h5z.cpid, minbits, maxbits, maxprec, minexp);
    else if (zfpmode == H5Z_ZFP_MODE_REVERSIBLE)
        H5Pset_zfp_reversible(h5z.cpid);

    hid_t ret_val;
    
    htri_t avail;
    unsigned filter_config;
    /*
    * Check that filter is registered with the library now.
    * If it is registered, retrieve filter's configuration.
    */
    avail = H5Zfilter_avail(H5Z_FILTER_ZFP);
    if (avail) {
        status = H5Zget_filter_info (H5Z_FILTER_ZFP, &filter_config);
        if ( (filter_config & H5Z_FILTER_CONFIG_ENCODE_ENABLED) &&
        (filter_config & H5Z_FILTER_CONFIG_DECODE_ENABLED) )
            printf ("filter is available for encoding and decoding.\n");
    }

    status = H5Pset_filter(h5z.cpid, H5Z_FILTER_ZFP, H5Z_FLAG_MANDATORY, cd_nelmts, cd_values);
    return;
}

static H5ZType createHDF5File(std::string fileName, MPI_Comm comm)
{
    H5ZType h5z;
    herr_t status;
    hid_t  fapl, fcpl;
    herr_t ret;

    // Enable the file to be accessible from multiple ranks
    fcpl = H5Pcreate(H5P_FILE_CREATE);
    fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl, comm, MPI_INFO_NULL);
    H5Pset_all_coll_metadata_ops(fapl, 1);
    H5Pset_coll_metadata_write(fapl, 1);
    // H5Pset_alignment(fapl,
    //                     1, //alignment_increment
    //                     0); //alignment_threshold

    h5z.file_id = H5Fcreate(fileName.c_str(), H5F_ACC_TRUNC, fcpl, fapl);
    ret = H5Pclose(fapl);
    return h5z;
}

//! @brief Open in parallel mode if supported, otherwise serial if numRanks == 1
static H5ZType openHDF5File(std::string fileName, MPI_Comm comm)
{
    H5ZType h5z;
    herr_t status;
    hid_t  fapl, fcpl;
    hid_t file_id;
    herr_t ret;

    // Enable the file to be accessible from multiple ranks
    fcpl = H5Pcreate(H5P_FILE_CREATE);
    fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl, comm, MPI_INFO_NULL);
    H5Pset_all_coll_metadata_ops(fapl, 1);
    H5Pset_coll_metadata_write(fapl, 1);
    // H5Pset_alignment(fapl,
    //                     1, //alignment_increment
    //                     0); //alignment_threshold

    h5z.file_id = H5Fopen(fileName.c_str(), H5F_ACC_RDWR, fapl);
    ret = H5Pclose(fapl);
    return h5z;
}

// When reading from an existing file, there's no need to close group.
static void closeHDF5File(H5ZType& h5z, bool reading = false) {
    if (!reading) {
        h5z.status = H5Gclose(h5z.group_id);
    }
    h5z.status = H5Fclose(h5z.file_id);
}

static void addHDF5Filter(H5ZType& h5z) {
    herr_t          status;
    hsize_t chunk_dims[2] ={2000, 1};
    h5z.cpid = H5Pcreate(H5P_DATASET_CREATE);
    status = H5Pset_chunk(h5z.cpid, 2, chunk_dims);
    
    if (h5z.compression == CompressionMethod::zfp) {
        int zfpmode = H5Z_ZFP_MODE_RATE;
        double rate = 4;
        double acc = 0;
        unsigned int prec = 11;
        unsigned int minbits = 0;
        unsigned int maxbits = 4171;
        unsigned int maxprec = 64;
        int minexp = -1074;
        setupZFP(h5z, zfpmode, rate, acc, prec, minbits, maxbits, maxprec, minexp);
    }
    if (h5z.compression == CompressionMethod::gzip) {
        status = H5Pset_deflate(h5z.cpid, h5z.compressionParam);
    }
    if (h5z.compression == CompressionMethod::szip) {
        unsigned szip_options_mask;
        unsigned szip_pixels_per_block;

        szip_options_mask = H5_SZIP_NN_OPTION_MASK;
        szip_pixels_per_block = h5z.compressionParam;
        status = H5Pset_szip(h5z.cpid, szip_options_mask, szip_pixels_per_block);
    }
}

static void addHDF5Step(H5ZType& h5z, std::string fieldName) {
    h5z.group_id = H5Gcreate2(h5z.file_id, fieldName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
}

//! @brief numParticles: total number of global particles
static void writeHDF5Field_(H5ZType& h5z, const std::string& fieldName, const void* field, hid_t dataType, uint64_t firstIndex=0, uint64_t lastIndex=0, uint64_t numParticles = 0, size_t nCol = 1) {
    // Following previous conventions, each field is written into a separate dataset.
    // Also, maxdim is set to exactly the data size + 1...for now
    hsize_t dims[2] = {numParticles, nCol};
    // hsize_t maxdims[2] = {H5S_UNLIMITED, H5S_UNLIMITED};
    h5z.dspace_id = H5Screate_simple(2, dims, NULL);
    addHDF5Filter(h5z);
 
    // Create dataset
    hid_t lcpl = H5Pcreate(H5P_LINK_CREATE);
    h5z.dset_id = H5Dcreate(h5z.group_id, fieldName.c_str(), dataType, h5z.dspace_id, lcpl, h5z.cpid, H5P_DEFAULT);

    // ======================
    // When writing, each process has an independent hyperslab
    /* set up dimensions of the slab this process accesses */
    // Dim0: numParticles, Dim1: numAttributes (unknown)
    // Count: dim0: numCurrParticles, dim1: numAttributes (unknown)
    // stride: numRanks, 1
    // block: 1, 1
    // Selection should be within dimension
    hid_t dataSpace = H5Dget_space(h5z.dset_id);
    hsize_t offset[2] = {firstIndex, 0};
    hsize_t count[2] = {lastIndex - firstIndex, nCol};
    hsize_t stride[2] = {1, 1};
    h5z.status = H5Sselect_hyperslab(dataSpace, H5S_SELECT_SET, offset, stride, count, NULL);

    /* create a memory dataspace independently */
    hid_t memorySpace = H5Screate_simple(1, count, NULL);

    // Write into dataspace independently
    hid_t dxpl = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio (dxpl, H5FD_MPIO_COLLECTIVE);

    MPI_Barrier(MPI_COMM_WORLD);

    h5z.status = H5Dwrite(h5z.dset_id, dataType, memorySpace, dataSpace, dxpl, field);
    // ======================

    if (h5z.compression == CompressionMethod::zfp) {
        H5Z_zfp_finalize();
    }

    h5z.status = H5Pclose(lcpl);
    h5z.status = H5Pclose(dxpl);
    h5z.status = H5Pclose(h5z.cpid);
    h5z.status = H5Sclose(memorySpace);
    h5z.status = H5Sclose(h5z.dspace_id);
    h5z.status = H5Dclose(h5z.dset_id);
}

static void writeHDF5Field(H5ZType& h5z, const std::string& fieldName, const float* field, uint64_t firstIndex, uint64_t lastIndex, uint64_t numParticles) {
    writeHDF5Field_(h5z, fieldName, (const void*)field, H5T_NATIVE_FLOAT, firstIndex, lastIndex, numParticles);
}

static void writeHDF5Field(H5ZType& h5z, const std::string& fieldName, const double* field, uint64_t firstIndex, uint64_t lastIndex, uint64_t numParticles) {
    writeHDF5Field_(h5z, fieldName, (const void*)field, H5T_NATIVE_DOUBLE, firstIndex, lastIndex, numParticles);
}

static void writeHDF5Field(H5ZType& h5z, const std::string& fieldName, const char* field, uint64_t firstIndex, uint64_t lastIndex, uint64_t numParticles) {
    writeHDF5Field_(h5z, fieldName, (const void*)field, H5T_NATIVE_CHAR, firstIndex, lastIndex, numParticles);
}

static void writeHDF5Field(H5ZType& h5z, const std::string& fieldName, const int* field, uint64_t firstIndex, uint64_t lastIndex, uint64_t numParticles) {
    writeHDF5Field_(h5z, fieldName, (const void*)field, H5T_NATIVE_INT32, firstIndex, lastIndex, numParticles);
}

static void writeHDF5Field(H5ZType& h5z, const std::string& fieldName, const unsigned int* field, uint64_t firstIndex, uint64_t lastIndex, uint64_t numParticles) {
    writeHDF5Field_(h5z, fieldName, (const void*)field, H5T_NATIVE_UINT32, firstIndex, lastIndex, numParticles);
}

static void writeHDF5Field(H5ZType& h5z, const std::string& fieldName, const long int* field, uint64_t firstIndex, uint64_t lastIndex, uint64_t numParticles) {
    writeHDF5Field_(h5z, fieldName, (const void*)field, H5T_NATIVE_LONG, firstIndex, lastIndex, numParticles);
}

static void writeHDF5Field(H5ZType& h5z, const std::string& fieldName, const long unsigned int* field, uint64_t firstIndex, uint64_t lastIndex, uint64_t numParticles) {
    writeHDF5Field_(h5z, fieldName, (const void*)field, H5T_NATIVE_ULONG, firstIndex, lastIndex, numParticles);
}

static void writeHDF5Attrib_(H5ZType& h5z, const std::string& fieldName, const void* field, size_t numElements, hid_t dataType) {
    hid_t attr_space_id = H5Screate(H5S_SIMPLE);
    hsize_t dim[1] = {numElements};
    herr_t ret  = H5Sset_extent_simple(attr_space_id, 1, dim, NULL);
    hid_t attrib_id = H5Acreate2(
        h5z.group_id,
        fieldName.c_str(),
        dataType,
        attr_space_id,
        H5P_DEFAULT,
        H5P_DEFAULT
    );
    H5Awrite(attrib_id, dataType, field);
    H5Sclose(attr_space_id);
    H5Aclose(attrib_id);
}

static void writeHDF5Attrib(H5ZType& h5z, const std::string& fieldName, const float* field, size_t numElements) {
    writeHDF5Attrib_(h5z, fieldName, (const void*)field, numElements, H5T_NATIVE_FLOAT);
}

static void writeHDF5Attrib(H5ZType& h5z, const std::string& fieldName, const double* field, size_t numElements) {
    writeHDF5Attrib_(h5z, fieldName, (const void*)field, numElements, H5T_NATIVE_DOUBLE);
}

static void writeHDF5Attrib(H5ZType& h5z, const std::string& fieldName, const char* field, size_t numElements) {
    writeHDF5Attrib_(h5z, fieldName, (const void*)field, numElements, H5T_NATIVE_CHAR);
}

static void writeHDF5Attrib(H5ZType& h5z, const std::string& fieldName, const int* field, size_t numElements) {
    writeHDF5Attrib_(h5z, fieldName, (const void*)field, numElements, H5T_NATIVE_INT32);
}

static void writeHDF5Attrib(H5ZType& h5z, const std::string& fieldName, const unsigned int* field, size_t numElements) {
    writeHDF5Attrib_(h5z, fieldName, (const void*)field, numElements, H5T_NATIVE_UINT32);
}

static void writeHDF5Attrib(H5ZType& h5z, const std::string& fieldName, const long int* field, size_t numElements) {
    writeHDF5Attrib_(h5z, fieldName, (const void*)field, numElements, H5T_NATIVE_LONG);
}

static void writeHDF5Attrib(H5ZType& h5z, const std::string& fieldName, const long unsigned int* field, size_t numElements) {
    writeHDF5Attrib_(h5z, fieldName, (const void*)field, numElements, H5T_NATIVE_ULONG);
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

template<class T>
auto sphexaWriteStepAttrib(H5PartFile* h5_file, const std::string& name, const T* value, size_t numElements)
{
    return H5PartWriteStepAttrib(h5_file, name.c_str(), H5PartType<T>{}, value, numElements);
}

template<class T>
auto sphexaWriteFileAttrib(H5PartFile* h5_file, const std::string& name, const T* value, size_t numElements)
{
    return H5PartWriteFileAttrib(h5_file, name.c_str(), H5PartType<T>{}, value, numElements);
}

/* read fields */

//! @brief numParticles: total number of global particles
static unsigned readHDF5Field_(H5ZType& h5z, const std::string& fieldName, void* field, hid_t dataType) {
    size_t nelmts      = 0;
    unsigned filter_info;
    unsigned flags;

    std::string datasetPath = "/Step#" + std::to_string(h5z.step) + "/" + fieldName;
    h5z.dset_id = H5Dopen2(h5z.file_id, datasetPath.c_str(), H5P_DEFAULT);

    /* Retrieve filter information. */
    hid_t plist_id = H5Dget_create_plist(h5z.dset_id);

    uint16_t numfilt = H5Pget_nfilters(plist_id);
    // printf("Number of filters associated with dataset: %i\n", numfilt);

    // for (uint16_t i = 0; i < numfilt; i++) {
    //     H5Z_filter_t filter_type = H5Pget_filter2(plist_id, i, &flags, &nelmts, NULL, 0, NULL, &filter_info);
    //     printf("Filter Type: ");
    //     switch (filter_type) {
    //         case H5Z_FILTER_DEFLATE:
    //             printf("H5Z_FILTER_DEFLATE\n");
    //             break;
    //         case H5Z_FILTER_SZIP:
    //             printf("H5Z_FILTER_SZIP\n");
    //             break;
    //         default:
    //             printf("Other filter type included.\n");
    //     }
    // }

    h5z.status = H5Dread(h5z.dset_id, dataType, H5S_ALL, H5S_ALL, H5P_DEFAULT, field);
    h5z.status = H5Dclose(h5z.dset_id);
    h5z.status = H5Pclose(plist_id);
    // h5z.status = H5Fclose(h5z.file_id);

    return 0;
}

inline unsigned readHDF5Field(H5ZType& h5z, const std::string& fieldName, double* field)
{
    return readHDF5Field_(h5z, fieldName, field, H5T_NATIVE_DOUBLE);
}

inline unsigned readHDF5Field(H5ZType& h5z, const std::string& fieldName, float* field)
{
    return readHDF5Field_(h5z, fieldName, field, H5T_NATIVE_DOUBLE);
}
inline unsigned readHDF5Field(H5ZType& h5z, const std::string& fieldName, char* field)
{
    return readHDF5Field_(h5z, fieldName, field, H5T_NATIVE_DOUBLE);
}
inline unsigned readHDF5Field(H5ZType& h5z, const std::string& fieldName, int* field)
{
    return readHDF5Field_(h5z, fieldName, field, H5T_NATIVE_DOUBLE);
}
inline unsigned readHDF5Field(H5ZType& h5z, const std::string& fieldName, int64_t* field)
{
    return readHDF5Field_(h5z, fieldName, field, H5T_NATIVE_DOUBLE);
}
inline unsigned readHDF5Field(H5ZType& h5z, const std::string& fieldName, unsigned* field)
{
    return readHDF5Field_(h5z, fieldName, field, H5T_NATIVE_DOUBLE);
}
inline unsigned readHDF5Field(H5ZType& h5z, const std::string& fieldName, uint64_t* field)
{
    return readHDF5Field_(h5z, fieldName, field, H5T_NATIVE_DOUBLE);
}
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
