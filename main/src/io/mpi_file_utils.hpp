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

enum class CompressionMethod
{
    none,
    deflate,
    gzip,
    szip,
    zfp
};

struct H5ZType
{
    CompressionMethod compression = CompressionMethod::none;
    int               compressionParam = 0;
    hid_t             file_id = 0;
    hid_t             group_id = 0;
    hid_t             dset_id = 0;
    hid_t             dspace_id = 0;
    hid_t             cpid = 0;
    hid_t             status = 0;
    hsize_t           numParticles = 0; // reading: num of local particles
    hsize_t           step = 0;         // only used in reading
    hsize_t           start = 0;        // only used in reading
};

static void setupZFP(H5ZType& h5z, int zfpmode, double rate, double acc, unsigned int prec, unsigned int minbits,
                     unsigned int maxbits, unsigned int maxprec, int minexp)
{
    hid_t        status;
    unsigned int cd_values[10];
    size_t       cd_nelmts = 10;

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

    htri_t   avail;
    unsigned filter_config;
    /*
     * Check that filter is registered with the library now.
     * If it is registered, retrieve filter's configuration.
     */
    avail = H5Zfilter_avail(H5Z_FILTER_ZFP);
    if (avail)
    {
        status = H5Zget_filter_info(H5Z_FILTER_ZFP, &filter_config);
        if ((filter_config & H5Z_FILTER_CONFIG_ENCODE_ENABLED) && (filter_config & H5Z_FILTER_CONFIG_DECODE_ENABLED))
            printf("filter is available for encoding and decoding.\n");
    }

    status = H5Pset_filter(h5z.cpid, H5Z_FILTER_ZFP, H5Z_FLAG_MANDATORY, cd_nelmts, cd_values);
    return;
}

static H5ZType createHDF5File(std::string fileName, MPI_Comm comm)
{
    H5ZType h5z;
    herr_t  status;
    hid_t   fapl, fcpl;
    herr_t  ret;

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
    ret         = H5Pclose(fapl);
    return h5z;
}

//! @brief Open in parallel mode if supported, otherwise serial if numRanks == 1
static H5ZType openHDF5File(std::string fileName, MPI_Comm comm)
{
    H5ZType h5z;
    herr_t  status;
    hid_t   fapl, fcpl;
    hid_t   file_id;
    herr_t  ret;

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
    ret         = H5Pclose(fapl);
    return h5z;
}

// When reading from an existing file, there's no need to close group.
static void closeHDF5File(H5ZType& h5z, bool reading = false)
{
    if (!reading && (h5z.group_id)) { 
        h5z.status = H5Gclose(h5z.group_id);
    }
    if (h5z.file_id) {
        h5z.status = H5Fclose(h5z.file_id);
    }
    h5z = H5ZType();
}

static void addHDF5Filter(H5ZType& h5z)
{
    herr_t  status;
    hsize_t chunk_dims[1] = {2000};
    h5z.cpid              = H5Pcreate(H5P_DATASET_CREATE);
    status                = H5Pset_chunk(h5z.cpid, 1, chunk_dims);

    if (h5z.compression == CompressionMethod::zfp)
    {
        int          zfpmode = H5Z_ZFP_MODE_RATE;
        double       rate    = 4;
        double       acc     = 0;
        unsigned int prec    = 11;
        unsigned int minbits = 0;
        unsigned int maxbits = 4171;
        unsigned int maxprec = 64;
        int          minexp  = -1074;
        setupZFP(h5z, zfpmode, rate, acc, prec, minbits, maxbits, maxprec, minexp);
    }
    if (h5z.compression == CompressionMethod::gzip) { status = H5Pset_deflate(h5z.cpid, h5z.compressionParam); }
    if (h5z.compression == CompressionMethod::szip)
    {
        unsigned szip_options_mask;
        unsigned szip_pixels_per_block;

        szip_options_mask     = H5_SZIP_NN_OPTION_MASK;
        szip_pixels_per_block = h5z.compressionParam;
        status                = H5Pset_szip(h5z.cpid, szip_options_mask, szip_pixels_per_block);
    }
}

static void addHDF5Step(H5ZType& h5z, std::string fieldName)
{
    h5z.group_id = H5Gcreate2(h5z.file_id, fieldName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
}

static void writeHDF5Attribute(H5ZType& h5z, std::string fieldName, const void* value, hid_t dataType, const hsize_t numElements)
{
    hid_t space_id = H5Screate_simple ( 1, &numElements, NULL );
    hid_t attrib_id = H5Acreate(
        h5z.group_id,
        fieldName.c_str(),
        dataType,
        space_id,
        H5P_DEFAULT
#ifndef H5_USE_16_API
        , H5P_DEFAULT
#endif
    );

    herr_t herr = H5Awrite ( attrib_id, dataType, value);
    herr = H5Aclose ( attrib_id );
    herr = H5Sclose ( space_id );
}

// Only difference between the following and writeHDF5Attribute:
// writeHDF5FileAttribute writes to root path of file, while writeHDF5Attribute writes to a group
static void writeHDF5FileAttribute(H5ZType& h5z, std::string fieldName, const void* value, hid_t dataType, const hsize_t numElements)
{
    hid_t space_id = H5Screate_simple ( 1, &numElements, NULL );
    hid_t attrib_id = H5Acreate(
        h5z.file_id,
        fieldName.c_str(),
        dataType,
        space_id,
        H5P_DEFAULT
#ifndef H5_USE_16_API
        , H5P_DEFAULT
#endif
    );

    herr_t herr = H5Awrite ( attrib_id, dataType, value);
    herr = H5Aclose ( attrib_id );
    herr = H5Sclose ( space_id );
}

//! @brief numParticles: total number of global particles
// firstIndex: start of local index
// lastIndex: end of local index
static void writeHDF5Field_(H5ZType& h5z, const std::string& fieldName, const void* field, hid_t dataType,
                            uint64_t firstIndex = 0, uint64_t lastIndex = 0, uint64_t numParticles = 0, size_t nCol = 1)
{
    // Following previous conventions, each field is written into a separate dataset.
    // Also, maxdim is set to exactly the data size + 1...for now
    hsize_t dims[1] = {numParticles};
    // hsize_t maxdims[2] = {H5S_UNLIMITED, H5S_UNLIMITED};
    h5z.dspace_id = H5Screate_simple(1, dims, NULL);
    addHDF5Filter(h5z);

    // Create dataset enough for all global particles
    hid_t lcpl  = H5Pcreate(H5P_LINK_CREATE);
    h5z.dset_id = H5Dcreate(h5z.group_id, fieldName.c_str(), dataType, h5z.dspace_id, lcpl, h5z.cpid, H5P_DEFAULT);

    // ======================
    // When writing, each process has an independent hyperslab
    /* set up dimensions of the slab this process accesses */
    // Dim0: numParticles, Dim1: numAttributes (unknown)
    // Count: dim0: numCurrParticles, dim1: numAttributes (unknown)
    // stride: numRanks, 1
    // block: 1, 1
    // Selection should be within dimension
    hid_t   dataSpace = H5Dget_space(h5z.dset_id);
    hsize_t offset[1] = {firstIndex};
    hsize_t count[1]  = {lastIndex - firstIndex};
    hsize_t stride[1] = {1};
    // Select a hyperslab for local particles (only with local size)
    h5z.status = H5Sselect_hyperslab(dataSpace, H5S_SELECT_SET, offset, stride, count, NULL);

    // create a memory dataspace independently, but only for local particles
    hid_t memorySpace = H5Screate_simple(1, count, NULL);

    // Write into dataspace independently
    hid_t dxpl = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE);

    MPI_Barrier(MPI_COMM_WORLD);

    h5z.status = H5Dwrite(h5z.dset_id, dataType, memorySpace, dataSpace, dxpl, field);
    // ======================

    if (h5z.compression == CompressionMethod::zfp) { H5Z_zfp_finalize(); }

    h5z.status = H5Pclose(lcpl);
    h5z.status = H5Pclose(dxpl);
    h5z.status = H5Pclose(h5z.cpid);
    h5z.status = H5Sclose(memorySpace);
    h5z.status = H5Sclose(h5z.dspace_id);
    h5z.status = H5Dclose(h5z.dset_id);
}

static void writeHDF5Field(H5ZType& h5z, const std::string& fieldName, const float* field, uint64_t firstIndex,
                           uint64_t lastIndex, uint64_t numParticles)
{
    writeHDF5Field_(h5z, fieldName, (const void*)field, H5T_NATIVE_FLOAT, firstIndex, lastIndex, numParticles);
}

static void writeHDF5Field(H5ZType& h5z, const std::string& fieldName, const double* field, uint64_t firstIndex,
                           uint64_t lastIndex, uint64_t numParticles)
{
    writeHDF5Field_(h5z, fieldName, (const void*)field, H5T_NATIVE_DOUBLE, firstIndex, lastIndex, numParticles);
}

static void writeHDF5Field(H5ZType& h5z, const std::string& fieldName, const char* field, uint64_t firstIndex,
                           uint64_t lastIndex, uint64_t numParticles)
{
    writeHDF5Field_(h5z, fieldName, (const void*)field, H5T_NATIVE_CHAR, firstIndex, lastIndex, numParticles);
}

static void writeHDF5Field(H5ZType& h5z, const std::string& fieldName, const int* field, uint64_t firstIndex,
                           uint64_t lastIndex, uint64_t numParticles)
{
    writeHDF5Field_(h5z, fieldName, (const void*)field, H5T_NATIVE_INT32, firstIndex, lastIndex, numParticles);
}

static void writeHDF5Field(H5ZType& h5z, const std::string& fieldName, const unsigned int* field, uint64_t firstIndex,
                           uint64_t lastIndex, uint64_t numParticles)
{
    writeHDF5Field_(h5z, fieldName, (const void*)field, H5T_NATIVE_UINT32, firstIndex, lastIndex, numParticles);
}

static void writeHDF5Field(H5ZType& h5z, const std::string& fieldName, const long int* field, uint64_t firstIndex,
                           uint64_t lastIndex, uint64_t numParticles)
{
    writeHDF5Field_(h5z, fieldName, (const void*)field, H5T_NATIVE_LONG, firstIndex, lastIndex, numParticles);
}

static void writeHDF5Field(H5ZType& h5z, const std::string& fieldName, const long unsigned int* field,
                           uint64_t firstIndex, uint64_t lastIndex, uint64_t numParticles)
{
    writeHDF5Field_(h5z, fieldName, (const void*)field, H5T_NATIVE_ULONG, firstIndex, lastIndex, numParticles);
}

static void writeHDF5Attrib_(H5ZType& h5z, const std::string& fieldName, const void* field, size_t numElements,
                             hid_t dataType)
{
    hid_t   attr_space_id = H5Screate(H5S_SIMPLE);
    hsize_t dim[1]        = {numElements};
    herr_t  ret           = H5Sset_extent_simple(attr_space_id, 1, dim, NULL);
    hid_t   attrib_id = H5Acreate2(h5z.group_id, fieldName.c_str(), dataType, attr_space_id, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attrib_id, dataType, field);
    H5Sclose(attr_space_id);
    H5Aclose(attrib_id);
}

static void writeHDF5Attrib(H5ZType& h5z, const std::string& fieldName, const float* field, size_t numElements)
{
    writeHDF5Attrib_(h5z, fieldName, (const void*)field, numElements, H5T_NATIVE_FLOAT);
}

static void writeHDF5Attrib(H5ZType& h5z, const std::string& fieldName, const double* field, size_t numElements)
{
    writeHDF5Attrib_(h5z, fieldName, (const void*)field, numElements, H5T_NATIVE_DOUBLE);
}

static void writeHDF5Attrib(H5ZType& h5z, const std::string& fieldName, const char* field, size_t numElements)
{
    writeHDF5Attrib_(h5z, fieldName, (const void*)field, numElements, H5T_NATIVE_CHAR);
}

static void writeHDF5Attrib(H5ZType& h5z, const std::string& fieldName, const int* field, size_t numElements)
{
    writeHDF5Attrib_(h5z, fieldName, (const void*)field, numElements, H5T_NATIVE_INT32);
}

static void writeHDF5Attrib(H5ZType& h5z, const std::string& fieldName, const unsigned int* field, size_t numElements)
{
    writeHDF5Attrib_(h5z, fieldName, (const void*)field, numElements, H5T_NATIVE_UINT32);
}

static void writeHDF5Attrib(H5ZType& h5z, const std::string& fieldName, const long int* field, size_t numElements)
{
    writeHDF5Attrib_(h5z, fieldName, (const void*)field, numElements, H5T_NATIVE_LONG);
}

static void writeHDF5Attrib(H5ZType& h5z, const std::string& fieldName, const long unsigned int* field,
                            size_t numElements)
{
    writeHDF5Attrib_(h5z, fieldName, (const void*)field, numElements, H5T_NATIVE_ULONG);
}


/* read fields */

//! @brief numParticles: total number of global particles
static unsigned readHDF5Field_(H5ZType& h5z, const std::string& fieldName, void* field, hid_t dataType,
                               uint64_t numParticles)
{
    size_t   nelmts = 0;
    unsigned filter_info;
    unsigned flags;

    // Memory space
    hsize_t dims[1] = {numParticles};
    h5z.dspace_id   = H5Screate_simple(1, dims, NULL);

    // Open the dataset
    std::string datasetPath = "/Step#" + std::to_string(h5z.step) + "/" + fieldName;
    h5z.dset_id             = H5Dopen1(h5z.file_id, datasetPath.c_str());

    // File space
    hid_t fileSpace = H5Dget_space(h5z.dset_id);

    // Retrieve filter information
    hid_t plist_id = H5Dget_create_plist(h5z.dset_id);
    uint16_t numfilt = H5Pget_nfilters(plist_id);
    nelmts = 0;
    H5Z_filter_t filter_type = H5Pget_filter (plist_id, 0, &flags, &nelmts, NULL, 0, NULL,
                &filter_info);
    printf ("Filter type is: ");
    switch (filter_type) {
        case H5Z_FILTER_DEFLATE:
            printf ("H5Z_FILTER_DEFLATE\n");
            break;
        case H5Z_FILTER_SHUFFLE:
            printf ("H5Z_FILTER_SHUFFLE\n");
            break;
        case H5Z_FILTER_FLETCHER32:
            printf ("H5Z_FILTER_FLETCHER32\n");
            break;
        case H5Z_FILTER_SZIP:
            printf ("H5Z_FILTER_SZIP\n");
            break;
        case H5Z_FILTER_NBIT:
            printf ("H5Z_FILTER_NBIT\n");
            break;
        case H5Z_FILTER_SCALEOFFSET:
            printf ("H5Z_FILTER_SCALEOFFSET\n");
    }
    hssize_t hh = H5Sget_simple_extent_npoints(fileSpace);

    // Select a hyperslab for local particles (only with local size)
    hsize_t stride[1] = {1};
    h5z.status        = H5Sselect_hyperslab(fileSpace, H5S_SELECT_SET, &h5z.start, NULL, &h5z.numParticles, NULL);

    // Read into dataspace independently
    // hid_t dxpl = H5Pcreate(H5P_DATASET_XFER);
    // H5Pset_dxpl_mpio (dxpl, H5FD_MPIO_COLLECTIVE);

    // MPI_Barrier(MPI_COMM_WORLD);

    // h5z.status = H5Dwrite(h5z.dset_id, dataType, memorySpace, dataSpace, dxpl, field);
    h5z.status = H5Dread(h5z.dset_id, dataType, h5z.dspace_id, H5S_ALL, H5P_DEFAULT, field);

    h5z.status = H5Dclose(h5z.dset_id);
    h5z.status = H5Sclose(h5z.dspace_id);
    h5z.status = H5Sclose(fileSpace);
    // h5z.status = H5Pclose(plist_id);
    return 0;
}

inline unsigned readHDF5Field(H5ZType& h5z, const std::string& fieldName, double* field, uint64_t numParticles)
{
    return readHDF5Field_(h5z, fieldName, field, H5T_NATIVE_DOUBLE, numParticles);
}
inline unsigned readHDF5Field(H5ZType& h5z, const std::string& fieldName, float* field, uint64_t numParticles)
{
    return readHDF5Field_(h5z, fieldName, field, H5T_NATIVE_FLOAT, numParticles);
}
inline unsigned readHDF5Field(H5ZType& h5z, const std::string& fieldName, char* field, uint64_t numParticles)
{
    return readHDF5Field_(h5z, fieldName, field, H5T_NATIVE_CHAR, numParticles);
}
inline unsigned readHDF5Field(H5ZType& h5z, const std::string& fieldName, int* field, uint64_t numParticles)
{
    return readHDF5Field_(h5z, fieldName, field, H5T_NATIVE_INT32, numParticles);
}
inline unsigned readHDF5Field(H5ZType& h5z, const std::string& fieldName, int64_t* field, uint64_t numParticles)
{
    return readHDF5Field_(h5z, fieldName, field, H5T_NATIVE_INT64, numParticles);
}
inline unsigned readHDF5Field(H5ZType& h5z, const std::string& fieldName, unsigned* field, uint64_t numParticles)
{
    return readHDF5Field_(h5z, fieldName, field, H5T_NATIVE_UINT32, numParticles);
}
inline unsigned readHDF5Field(H5ZType& h5z, const std::string& fieldName, uint64_t* field, uint64_t numParticles)
{
    return readHDF5Field_(h5z, fieldName, field, H5T_NATIVE_UINT64, numParticles);
}

} // namespace fileutils
} // namespace sphexa