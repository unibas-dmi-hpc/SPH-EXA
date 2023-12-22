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

template<class T>
size_t ADIOSGetNumParticles(ADIOS2Settings& as)
{
    // Global numParticles is written in numParticlesGlobal in step 0. Type is double.
    adios2::Variable<double> var = bpIO.InquireVariable<double>("numParticlesGlobal");
    double numParticles;
    bpReader.Get<double>(var, numParticles);
    return static_cast<int>(var);
}
template<class T>
size_t ADIOSGetNumSteps(ADIOS2Settings& as)
{
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