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

struct ADIOS2Settings {
    double accuracy = 1E-6;
    adios2::IO io;
    adios2::ADIOS adios;
    MPI_Comm comm;
    std::string fileName;

    size_t numLocalParticles;
    size_t numTotalRanks;
    std::string stepPrefix;
    size_t offset;
    size_t rank;
};

// Suppose we only have 1D arrays here
void initADIOSWriter(ADIOS2Settings & as) {
    /** ADIOS class factory of IO class objects */
    #if ADIOS2_USE_MPI
            as.adios = adios2::ADIOS(as.comm);
    #else
            as.adios = adios2::ADIOS();
    #endif
    as.io = as.adios.DeclareIO("BPFile_SZ");
    return;
}

template<class T>
void writeADIOSField(ADIOS2Settings & as, const std::string & fieldName, const T * field) {
    #if ADIOS2_USE_MPI
        adios2::ADIOS adios(MPI_COMM_WORLD);
    #else
        adios2::ADIOS adios;
    #endif
    as.io = adios.DeclareIO("BPFile_SZ");
    adios2::Variable<T> varDoubles = as.io.DefineVariable<T>(
        as.stepPrefix + fieldName, // Field name
        {as.numLocalParticles * as.numTotalRanks}, // Global dimensions
        {as.offset}, // Starting local offset
        {as.numLocalParticles},        // Local dimensions  (limited to 1 rank)
        adios2::ConstantDims);

    if (as.accuracy > 1E-16)
    {
        adios2::Operator op = adios.DefineOperator("SZCompressor", "sz");
        varDoubles.AddOperation(op, {{"accuracy", std::to_string(as.accuracy)}});
    }

    adios2::Attribute<double> attribute = as.io.DefineAttribute<double>("SZ_accuracy", as.accuracy);

    // To avoid compiling warnings
    (void)attribute;

    adios2::Engine bpWriter = as.io.Open(as.fileName, adios2::Mode::Append);
    bpWriter.BeginStep();
    bpWriter.Put(varDoubles, field);
    bpWriter.EndStep();
    bpWriter.Close();
}

template<class T>
void writeADIOSAttribute(ADIOS2Settings & as, const std::string & fieldName, const T * field) {
    #if ADIOS2_USE_MPI
        adios2::ADIOS adios(MPI_COMM_WORLD);
    #else
        adios2::ADIOS adios;
    #endif
    as.io = adios.DeclareIO("BPFile_SZ");
    if (as.rank == 0) {
        adios2::Variable<T> varAttrib = as.io.DefineVariable<T>(
            as.stepPrefix + fieldName // Field name
            );
        adios2::Engine bpWriter = as.io.Open(as.fileName, adios2::Mode::Append);
        bpWriter.BeginStep();
        bpWriter.Put( varAttrib, field );
        bpWriter.EndStep();
        bpWriter.Close();
    }
}


} // namespace fileutils
} // namespace sphexa