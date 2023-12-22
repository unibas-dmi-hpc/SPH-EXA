#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "cstone/util/type_list.hpp"
#include "cstone/util/tuple_util.hpp"
#include "adios2.h"
#include "adios_wrapper.hpp"
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

int64_t ADIOSGetNumParticles(ADIOS2Settings& as)
{
    // Global numParticles is written in numParticlesGlobal in step 0. Type is double.
    adios2::Attribute<double> var = as.io.InquireAttribute<double>("numParticlesGlobal");
    double                   numParticles;
    as.reader.Get<double>(var, numParticles);
    return static_cast<int64_t>(numParticles);
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

template<class T>
T readADIOSStepAttribute(ADIOS2Settings& as, const std::string& fieldName)
{
    // One MPI_COMM can only have one ADIOS instance.
    // Thus if we need another instance to write, has to recreate without the original as.comm.
    // Step attribute is a variable
    adios2::Variable<T> variable      = as.io.InquireVariable<T>(fieldName);
    variable.SetSelection({{0}, {1}});
    std::vector<T> res(1);
    as.reader.Get(variable, res.data(), adios2::Mode::Sync);
    return res[0];
}

template<class T>
T readADIOSFileAttribute(ADIOS2Settings& as, const std::string& fieldName)
{
    // One MPI_COMM can only have one ADIOS instance.
    // Thus if we need another instance to write, has to recreate without the original as.comm.
    // File attribute is a real attribute.
    T res      = as.io.InquireAttribute<T>(fieldName).Data()[0];
    return res;
}

} // namespace fileutils
} // namespace sphexa