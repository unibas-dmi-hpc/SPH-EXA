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
    size_t numTotalRanks     = 0;
    size_t offset            = 0;
    size_t rank              = 0;
    size_t currStep          = 0; // Step, not numIteration
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
    // Technically the var definition should be outside of BeginStep()/EndStep() loop
    // But given the current I/O APIs, and that numLocalParticles is not fixed,
    // I leave it like this.
    // Consequences are the "step" logic in ADIOS is not fully operating.
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
        // For now due to inconsistency of global attrib and step attrib, I set them all to double.
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
        // For now due to inconsistency of global attrib and step attrib, I set them all to double.
        as.io.DefineAttribute<double>(fieldName, (double)(field[0]));
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
    as.io = as.adios.DeclareIO("bpio");
    return;
}

void openADIOSStepRead(ADIOS2Settings& as) { as.reader = as.io.Open(as.fileName, adios2::Mode::ReadRandomAccess); }

void closeADIOSStepRead(ADIOS2Settings& as) { as.reader.Close(); }

int64_t ADIOSGetNumParticles(ADIOS2Settings& as)
{
    adios2::Variable<uint64_t> variable = as.io.InquireVariable<uint64_t>("numParticlesGlobal");
    variable.SetStepSelection({as.currStep - 1, 1});
    std::vector<uint64_t> varData(1);
    as.reader.Get(variable, varData.data(), adios2::Mode::Sync);
    return varData[0];
}

// This is step, not numIteration!!!
// Here the numSteps includes step0 (i.e. the step without actual particles)
// Returns the last Step where iteration is available
// Actual last step to read from should be {res.back() - 1, 1}
int64_t ADIOSGetNumSteps(ADIOS2Settings& as)
{
    // Time is a stepAttrib that gets written in all output formats
    // AllStepsBlocksInfo() is very expensive. Make sure it's only used once.
    adios2::Variable<uint64_t> variable = as.io.InquireVariable<uint64_t>("iteration");
    std::vector<size_t>        res      = as.reader.GetAbsoluteSteps(variable);
    if (res.back() <= 0) { throw std::runtime_error("The imported file is empty!"); }
    return res.back();
}

auto ADIOSGetFileAttributes(ADIOS2Settings& as)
{
    const std::map<std::string, adios2::Params> attribs = as.io.AvailableAttributes();
    std::vector<std::string>                    res;
    std::ranges::transform(attribs, std::back_inserter(res), [](const auto& pair) { return pair.first; });
    return res;
}

uint64_t ADIOSGetNumIterations(ADIOS2Settings& as)
{
    // Time is a stepAttrib that gets written in all output formats
    // AllStepsBlocksInfo() is very expensive. Make sure it's only used once.
    if (as.currStep == 0)
    {
        throw std::runtime_error(
            "The imported file is empty! ADIOSGetNumIterations() should be called after ADIOSGetNumSteps().");
    }
    adios2::Variable<uint64_t> variable = as.io.InquireVariable<uint64_t>("iteration");
    variable.SetStepSelection({as.currStep - 1, 1});
    // variable.SetSelection(adios2::Box<adios2::Dims>(0, 1));
    std::vector<uint64_t> res(1);
    as.reader.Get(variable, res.data(), adios2::Mode::Sync);
    return res[0];
}

template<class ExtractType>
void readADIOSField(ADIOS2Settings& as, const std::string& fieldName, ExtractType* field)
{
    if (as.currStep == 0)
    {
        throw std::runtime_error(
            "The imported file is empty! readADIOSField() should be called after ADIOSGetNumSteps().");
    }
    adios2::Variable<ExtractType> variable = as.io.InquireVariable<ExtractType>(fieldName);
    variable.SetStepSelection({as.currStep - 1, 1});
    variable.SetSelection({{as.offset}, {as.numLocalParticles}});
    as.reader.Get(variable, field, adios2::Mode::Sync);
}

template<class ExtractType>
void readADIOSStepAttribute(ADIOS2Settings& as, const std::string& fieldName, ExtractType* attr)
{
    if (as.currStep == 0)
    {
        throw std::runtime_error(
            "The imported file is empty! readADIOSStepAttribute() should be called after ADIOSGetNumSteps().");
    }
    // StepAttrib is an ADIOS single variable.
    // Should be read by only 1 rank.
    if (as.rank == 0)
    {
        adios2::Variable<ExtractType> variable = as.io.InquireVariable<ExtractType>(fieldName);
        variable.SetStepSelection({as.currStep - 1, 1});
        // variable.SetSelection(adios2::Box<adios2::Dims>(0, 1));
        as.reader.Get(variable, attr, adios2::Mode::Sync);
    }
}

template<class ExtractType>
void readADIOSFileAttribute(ADIOS2Settings& as, const std::string& fieldName, ExtractType* attr)
{
    // One MPI_COMM can only have one ADIOS instance.
    // Thus if we need another instance to write, has to recreate without the original as.comm.
    // File attribute is a real attribute.
    // Should be read by only 1 rank.
    if (as.rank == 0)
    {
        adios2::Attribute<ExtractType> res = as.io.InquireAttribute<ExtractType>(fieldName);
        attr[0]                            = res.Data()[0];
    }
}

} // namespace fileutils
} // namespace sphexa