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
    // General settings
    adios2::IO io;
    adios2::ADIOS adios;
    adios2::Operator adiosOp;
    MPI_Comm comm;
    std::string fileName;

    // Compression settings
    double accuracy = 1E-6;

    // Rank-specific settings
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
    as.adiosOp = as.adios.DefineOperator("SZCompressor", "sz");
    as.io = as.adios.DeclareIO("BPFile_SZ");
    adios2::Attribute<double> attribute = as.io.DefineAttribute<double>("SZ_accuracy", as.accuracy);
    // To avoid compiling warnings
    (void)attribute;
    return;
}

template<class T>
void writeADIOSField(ADIOS2Settings & as, const std::string & fieldName, const T * field) {
    adios2::Variable<T> varDoubles = as.io.DefineVariable<T>(
        as.stepPrefix + fieldName, // Field name
        {as.numLocalParticles * as.numTotalRanks}, // Global dimensions
        {as.offset}, // Starting local offset
        {as.numLocalParticles},        // Local dimensions  (limited to 1 rank)
        adios2::ConstantDims);

    if (as.accuracy > 1E-16)
    {
        varDoubles.AddOperation(as.adiosOp, {{"accuracy", std::to_string(as.accuracy)}});
    }

    adios2::Engine bpWriter = as.io.Open(as.fileName, adios2::Mode::Append);
    bpWriter.BeginStep();
    bpWriter.Put(varDoubles, field);
    bpWriter.EndStep();
    bpWriter.Close();
}

template<class T>
void writeADIOSStepAttribute(ADIOS2Settings & as, const std::string & fieldName, T *& field) {
    // std::cout << "write file attr in adios with rank " << as.rank << std::endl;
    // One MPI_COMM can only have one ADIOS instance.
    // Thus if we need another instance to write, has to recreate without the original as.comm.
    adios2::Engine bpWriter = as.io.Open(as.fileName, adios2::Mode::Append);
    as.io.DefineAttribute<T>(
        as.stepPrefix + fieldName, // Field name
        T()
        );
    // if (as.rank == 0) {
    // // #if ADIOS2_USE_MPI
    // //         adios2::ADIOS tempAdios = adios2::ADIOS(as.comm);
    // // #else
    // //         adios2::ADIOS tempAdios = adios2::ADIOS();
    // // #endif
    //     // adios2::ADIOS tempAdios = adios2::ADIOS();
    //     // adios2::IO tempIO = tempAdios.DeclareIO("BPFile_SZ");

    //     // adios2::Engine bpWriter = as.io.Open(as.fileName, adios2::Mode::Append);
    //     bpWriter.BeginStep();
    //     bpWriter.Put( varAttrib, field );
    //     bpWriter.EndStep();
    //     // bpWriter.Close();
    // }
    bpWriter.Close();
}

template<class T>
void writeADIOSFileAttribute(ADIOS2Settings & as, const std::string & fieldName, T *& field) {
    // std::cout << "write file attr in adios with rank " << as.rank << std::endl;
    // One MPI_COMM can only have one ADIOS instance.
    // Thus if we need another instance to write, has to recreate without the original as.comm.
    adios2::Engine bpWriter = as.io.Open(as.fileName, adios2::Mode::Append);
    as.io.DefineAttribute<T>(
        fieldName, // Field name
        T()
        );
    if (as.rank == 0) {
    // #if ADIOS2_USE_MPI
    //         adios2::ADIOS tempAdios = adios2::ADIOS(as.comm);
    // #else
    //         adios2::ADIOS tempAdios = adios2::ADIOS();
    // #endif
        // adios2::ADIOS tempAdios = adios2::ADIOS();
        // adios2::IO tempIO = tempAdios.DeclareIO("BPFile_SZ");

        // adios2::Engine bpWriter = as.io.Open(as.fileName, adios2::Mode::Append);
        bpWriter.BeginStep();
        // bpWriter.Put( varAttrib, field );
        bpWriter.EndStep();
        // bpWriter.Close();
    }
    bpWriter.Close();
}


} // namespace fileutils
} // namespace sphexa