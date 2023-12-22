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
    size_t numLocalParticles;
    size_t numTotalRanks;
    size_t offset;
    size_t rank;
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
    adios2::Attribute<double> attribute = as.io.DefineAttribute<double>("SZ_accuracy", as.accuracy);
    // To avoid compiling warnings
    (void)attribute;
    return;
}

// Suppose we only have 1D arrays here
void initADIOSReader(ADIOS2Settings& as)
{
#if ADIOS2_USE_MPI
    as.adios = adios2::ADIOS(as.comm);
#else
    as.adios = adios2::ADIOS();
#endif
    as.io                               = as.adios.DeclareIO("bpio");
    adios2::Attribute<double> attribute = as.io.DefineAttribute<double>("SZ_accuracy", as.accuracy);
    // To avoid compiling warnings
    (void)attribute;
    return;
}

template<class T>
size_t ADIOSGetNumSteps(ADIOS2Settings& as)
{
    return 0;
}

template<class T>
size_t ADIOSGetNumParticles(ADIOS2Settings& as)
{
    return 0;
}

void openADIOSStepRead(ADIOS2Settings& as)
{
    as.reader = as.io.Open(as.fileName, adios2::Mode::Read);
    as.reader.BeginStep();
}

void closeADIOSStepRead(ADIOS2Settings& as)
{
    as.reader.EndStep();
    as.reader.Close();
}

void openADIOSStepWrite(ADIOS2Settings& as)
{
    as.writer = as.io.Open(as.fileName, adios2::Mode::Append);
    as.writer.BeginStep();
}

void closeADIOSStepWrite(ADIOS2Settings& as)
{
    as.writer.EndStep();
    as.writer.Close();
}

template<class T>
void writeADIOSField(ADIOS2Settings& as, const std::string& fieldName, const T* field)
{
    adios2::Variable<T> var = as.io.DefineVariable<T>(fieldName,                                 // Field name
                                                      {as.numLocalParticles * as.numTotalRanks}, // Global dimensions
                                                      {as.offset},            // Starting local offset
                                                      {as.numLocalParticles}, // Local dimensions (limited to 1 rank)
                                                      adios2::ConstantDims);
    var.AddOperation("sz", {{"accuracy", std::to_string(as.accuracy)}});
    as.writer.Put(var, field);
}

template<class T>
void writeADIOSStepAttribute(ADIOS2Settings& as, const std::string& fieldName, const T* field)
{
    // StepAttribute is one variable with multiple copies (i.e. steps).
    // It's almost same as a field. Write only in rank 0.
    // There's no need for compression.
    if (as.rank == 0)
    {
        adios2::Variable<T> var = as.io.DefineVariable<T>(fieldName);
        as.writer.Put(var, field);
    }
}

template<class T>
void writeADIOSFileAttribute(ADIOS2Settings& as, const std::string& fieldName, const T* field)
{
    // FileAttribute is a unique variable. It is written into step 0.
    // Write only in rank 0.
    // There's no need for compression.
    if (as.rank == 0)
    {
        adios2::Variable<T> var = as.io.DefineVariable<T>(fieldName);
        as.writer.Put(var, field);
    }
}

template<class T>
void readADIOSField(ADIOS2Settings& as, const std::string& fieldName, const T* field)
{
    adios2::Variable<T> variable = as.io.InquireVariable<T>(fieldName);
    variable.SetSelection(adios2::Box<adios2::Dims>(as.offset,           // Offset
                                                    as.numLocalParticles // size to read
                                                    ));
    as.reader.Get(variable, field, adios2::Mode::Sync);
}

template<class T>
std::vector<T> readADIOSStepAttribute(ADIOS2Settings& as, const std::string& fieldName)
{
    // One MPI_COMM can only have one ADIOS instance.
    // Thus if we need another instance to write, has to recreate without the original as.comm.
    adios2::Engine bpReader = as.io.Open(as.fileName, adios2::Mode::Read);
    std::vector<T> res      = as.io.InquireAttribute<T>(fieldName).data();
    bpReader.Close();
    return res;
}

template<class T>
auto readADIOSFileAttribute(ADIOS2Settings& as, const std::string& fieldName)
{
    // One MPI_COMM can only have one ADIOS instance.
    // Thus if we need another instance to write, has to recreate without the original as.comm.
    adios2::Engine bpReader = as.io.Open(as.fileName, adios2::Mode::Read);
    std::vector<T> res      = as.io.InquireAttribute<T>(fieldName).data();
    bpReader.Close();
    return res;
}

} // namespace fileutils
} // namespace sphexa