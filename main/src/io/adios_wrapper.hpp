#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "cstone/util/type_list.hpp"
#include "cstone/util/tuple_util.hpp"
#include "adios2.h"
#if ADIOS2_USE_MPI
#include <mpi.h>
#endif

namespace sphexa
{
namespace fileutils
{

struct ADIOS2Settings
{
    // General settings
    adios2::IO       io;
    adios2::ADIOS    adios;
    adios2::Operator adiosOp;
    adios2::Engine   reader;
    adios2::Engine   writer;
    MPI_Comm         comm;
    std::string      fileName;

    // Compression settings
    double accuracy = 1E-6;

    // Rank-specific settings
    size_t numLocalParticles = 0;
    size_t numTotalRanks = 0;
    size_t offset = 0;
    size_t rank = 0;
    size_t currStep = 0;

    std::map<std::string, void*> fieldMap;
};

// One executable should have 1 adios instance only
void initADIOSWriter(ADIOS2Settings& as)
{
#if ADIOS2_USE_MPI
    as.adios = adios2::ADIOS(as.comm);
#else
    as.adios = adios2::ADIOS();
#endif
    as.io                               = as.adios.DeclareIO("bpio");
    as.fieldMap["SZ_accuracy"] = new adios2::Variable<double>(as.io.DefineVariable<double>("SZ_accuracy"));
    as.writer = as.io.Open(as.fileName, adios2::Mode::Append);
    return;
}

void closeADIOSWriter(ADIOS2Settings& as) {
    for (auto it = as.fieldMap.begin(); it != as.fieldMap.end(); ++it) {
        delete it->second; // Delete the pointer
    }
    as.fieldMap.clear();
    as.writer.Close();
}

void openADIOSStepWrite(ADIOS2Settings& as)
{
    as.writer.BeginStep();
}

void closeADIOSStepWrite(ADIOS2Settings& as)
{
    
    as.writer.EndStep();
}

template<class T>
void writeADIOSField(ADIOS2Settings& as, const std::string& fieldName, const T* field)
{
    auto it = as.fieldMap.find(fieldName);
    if (it == as.fieldMap.end()) {
        as.fieldMap[fieldName] = new adios2::Variable<T>(as.io.DefineVariable<T>(
        fieldName,                                 // Field name
        {as.numLocalParticles * as.numTotalRanks}, // Global dimensions
        {as.offset},            // Starting local offset
        {as.numLocalParticles}, // Local dimensions (limited to 1 rank)
        adios2::ConstantDims));
    }
    as.writer.Put(*(adios2::Variable<T> *)(as.fieldMap[fieldName]), field);
}

template<class T>
void writeADIOSStepAttribute(ADIOS2Settings& as, const std::string& fieldName, const T* field)
{
    // StepAttribute is one variable with multiple copies (i.e. steps).
    // It's almost same as a field. Write only in rank 0.
    // There's no need for compression.
    if (as.rank == 0)
    {
        // For now due to inconsistency of global attrib and step attrib, I set them all to double.
        as.io.DefineAttribute<double>(fieldName, (double)(field[0]), "", "/", true);
    }
}

template<class T>
void writeADIOSFileAttribute(ADIOS2Settings& as, const std::string& fieldName, const T * field)
{
    // FileAttribute is a unique variable. It is written into step 0.
    // Write only in rank 0.
    // There's no need for compression.
    if (as.rank == 0)
    {
        // For now due to inconsistency of global attrib and step attrib, I set them all to double.
        as.io.DefineAttribute<double>(fieldName, (double)(field[0]), "", "/", true);
    }
}

// ============================================================================
// ============================================================================

// Suppose we only have 1D arrays here
void initADIOSReader(ADIOS2Settings& as)
{
#if ADIOS2_USE_MPI
    as.adios = adios2::ADIOS(as.comm);
#else
    as.adios = adios2::ADIOS();
#endif
    as.io                               = as.adios.DeclareIO("bpio");
    // adios2::Attribute<double> attribute = as.io.DefineAttribute<double>("SZ_accuracy", as.accuracy);
    // To avoid compiling warnings
    // (void)attribute;
    return;
}
void openADIOSStepRead(ADIOS2Settings& as)
{
    as.reader = as.io.Open(as.fileName, adios2::Mode::ReadRandomAccess);
    as.reader.BeginStep();
}

void closeADIOSStepRead(ADIOS2Settings& as)
{
    as.reader.EndStep();
    as.reader.Close();
}

int64_t ADIOSGetNumParticles(ADIOS2Settings& as)
{
    // Global numParticles is written in numParticlesGlobal in step 0. Type is double.
    double res      = as.io.InquireAttribute<double>("numParticlesGlobal").Data()[0];
    return static_cast<int64_t>(res);
}

int ADIOSGetNumSteps(ADIOS2Settings& as)
{
    // When using this func, a new Engine has to be created
    // For step-wise access
    // adios2::Engine 
    return 0;
}

template<class T>
void readADIOSField(ADIOS2Settings& as, const std::string& fieldName, T* field)
{
    adios2::Variable<T> variable = as.io.InquireVariable<T>(fieldName);
    variable.SetSelection(adios2::Box<adios2::Dims>(as.offset,           // Offset
                                                    as.numLocalParticles // size to read
                                                    ));
    as.reader.Get(variable, field, adios2::Mode::Sync);
}

template<class ExtractType>
ExtractType readADIOSStepAttribute(ADIOS2Settings& as, const std::string& fieldName, ExtractType* attr)
{
    // One MPI_COMM can only have one ADIOS instance.
    // Thus if we need another instance to write, has to recreate without the original as.comm.
    // Step attribute is a variable
    adios2::Variable<ExtractType> variable      = as.io.InquireVariable<ExtractType>(fieldName);
    variable.SetSelection({{0}, {1}});
    std::vector<ExtractType> res(1);
    as.reader.Get(variable, res.data(), adios2::Mode::Sync);
    return res[0];
}

template<class ExtractType>
ExtractType readADIOSFileAttribute(ADIOS2Settings& as, const std::string& fieldName, ExtractType* attr)
{
    // One MPI_COMM can only have one ADIOS instance.
    // Thus if we need another instance to write, has to recreate without the original as.comm.
    // File attribute is a real attribute.
    // T res      = as.io.InquireAttribute<T>(fieldName)
    adios2::Attribute<ExtractType> res = as.io.InquireAttribute<ExtractType>(fieldName);
    return res.Data()[0];
    // return res;
}

} // namespace fileutils
} // namespace sphexa