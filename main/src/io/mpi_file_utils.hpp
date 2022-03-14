#pragma once

#ifdef SPH_EXA_HAVE_H5PART
#include <filesystem>
#include "H5Part.h"
#endif

namespace sphexa
{
namespace fileutils
{

#ifdef SPH_EXA_HAVE_H5PART

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

template<class Dataset, class T>
void writeH5Part(Dataset& d, size_t firstIndex, size_t lastIndex, const cstone::Box<T>& box, const std::string& path)
{
    using h5_int64_t = h5part_int64_t;
    using h5_id_t    = h5part_int64_t;

    // output name
    const char* h5_fname = path.c_str();
    H5PartFile* h5_file  = nullptr;

#ifdef H5PART_PARALLEL_IO
    if (std::filesystem::exists(h5_fname)) { h5_file = H5PartOpenFileParallel(h5_fname, H5PART_APPEND, d.comm); }
    else
    {
        h5_file = H5PartOpenFileParallel(h5_fname, H5PART_WRITE, d.comm);
    }
#else
    if (d.nrank > 1)
    {
        throw std::runtime_error("Cannot write HDF5 output with multiple ranks without parallel HDF5 support\n");
    }
    if (std::filesystem::exists(h5_fname)) { h5_file = H5PartOpenFile(h5_fname, H5PART_APPEND); }
    else
    {
        h5_file = H5PartOpenFile(h5_fname, H5PART_WRITE);
    }
#endif

    // create the next step
    h5_id_t numSteps = H5PartGetNumSteps(h5_file);
    H5PartSetStep(h5_file, numSteps);

    H5PartWriteStepAttrib(h5_file, "time", H5PART_FLOAT64, &d.ttot, 1);
    H5PartWriteStepAttrib(h5_file, "minDt", H5PART_FLOAT64, &d.minDt, 1);
    // record the actual SPH-iteration as step attribute
    H5PartWriteStepAttrib(h5_file, "step", H5PART_INT64, &d.iteration, 1);
    H5PartWriteStepAttrib(h5_file, "gravConstant", H5PART_FLOAT64, &d.g, 1);

    // record the global coordinate bounding box
    double extents[6] = {box.xmin(), box.xmax(), box.ymin(), box.ymax(), box.zmin(), box.zmax()};
    H5PartWriteStepAttrib(h5_file, "box", H5PART_FLOAT64, extents, 6);
    h5part_int32_t pbc[3] = {box.pbcX(), box.pbcY(), box.pbcZ()};
    H5PartWriteStepAttrib(h5_file, "pbc", H5PART_INT32, pbc, 3);

    const h5_int64_t h5_num_particles = lastIndex - firstIndex;
    // set number of particles that each rank will write
    H5PartSetNumParticles(h5_file, h5_num_particles);

    auto fieldPointers = getOutputArrays(d);
    for (size_t fidx = 0; fidx < fieldPointers.size(); ++fidx)
    {
        const std::string& fieldName = Dataset::fieldNames[d.outputFields[fidx]];
        writeH5PartField(h5_file, fieldName, fieldPointers[fidx] + firstIndex);
    }

    H5PartCloseFile(h5_file);
}

#endif

} // namespace fileutils
} // namespace sphexa
