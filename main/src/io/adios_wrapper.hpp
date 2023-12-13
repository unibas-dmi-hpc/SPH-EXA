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
    MPI_Comm comm;
    string & fileName;

    size_t numLocalParticles;
    size_t numRanks;
    size_t offset;
    size_t rank;
};

// Suppose we only have 1D arrays here
void initADIOSWriter(ADIOS2Settings & as, MPI_Comm comm, string & fileName, double accuracy = 1E-6, size_t numLocalParticles, size_t numRanks, size_t offset) {
    /** ADIOS class factory of IO class objects */
    #if ADIOS2_USE_MPI
            adios2::ADIOS adios(MPI_COMM_WORLD);
    #else
            adios2::ADIOS adios;
    #endif
    as.io = adios.DeclareIO("BPFile_SZ");
    as.fileName = fileName;
    as.accuracy = accuracy;
    as.numLocalParticles = numLocalParticles;
    as.numRanks = numRanks;
    as.offset = offset;
    return;
}


void writeADIOSField(ADIOS2Settings & as, string & fieldName, const double* field) {
    adios2::Variable<double> varDoubles = bpIO.DefineVariable<double>(
        fieldName, // Field name
        {as.numLocalParticles * as.numRanks}, // Global dimensions
        {as.offset}, // Starting local offset
        {as.numLocalParticles},        // Local dimensions  (limited to 1 rank)
        adios2::ConstantDims);

    if (as.accuracy > 1E-16)
    {
        adios2::Operator op = adios.DefineOperator("SZCompressor", "sz");
        varDoubles.AddOperation(op, {{"accuracy", std::to_string(as.accuracy)}});
    }

    adios2::Attribute<double> attribute = bpIO.DefineAttribute<double>("SZ_accuracy", as.accuracy);

    // To avoid compiling warnings
    (void)attribute;

    adios2::Engine bpWriter = bpIO.Open(as.fileName, adios2::Mode::Append);
    bpWriter.BeginStep();
    bpWriter.Put(varDoubles, field);
    bpWriter.EndStep();
    bpWriter.Close();
}


void writeADIOSAttribute(ADIOS2Settings & as, string & fieldName, const double & field) {
    if (as.rank == 0) {
        adios2::Variable<double> varAttrib = bpIO.DefineVariable<double>(
            fieldName, // Field name
            );
        adios2::Engine bpWriter = bpIO.Open(as.fileName, adios2::Mode::Append);
        bpWriter.BeginStep();
        engine.Put( varAttrib, field );
        bpWriter.EndStep();
        bpWriter.Close();
    }
}


} // namespace fileutils
} // namespace sphexa