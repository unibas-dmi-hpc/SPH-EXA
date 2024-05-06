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
/*! @brief Class for parsing and storing compressor settings
 *
 * Following fields are mandatory:
 * name
 */
class CompressorSettings
{
private:
    std::map<std::string, std::string> settingsMap_;
    std::string                        settingsString_;

public:
    CompressorSettings() {}

    CompressorSettings(const std::string& input, char pairDelimiter = ',', char keyValueDelimiter = '=')
        : settingsString_(input)
    {
        std::istringstream iss(input);
        std::string        token;

        while (std::getline(iss, token, pairDelimiter))
        {
            std::istringstream tokenStream(token);
            std::string        key, value;

            if (std::getline(tokenStream, key, keyValueDelimiter) && std::getline(tokenStream, value))
            {
                settingsMap_[key] = value;
            }
        }
    }

    CompressorSettings(const CompressorSettings& other) { settingsMap_ = other.settingsMap_; }

    void insert(const std::string& key, const std::string& value) { settingsMap_[key] = value; }

    std::string get(const std::string& key, const std::string defaultValue = "") const
    {
        auto it = settingsMap_.find(key);
        if (it != settingsMap_.end()) { return it->second; }
        return defaultValue; // Return an empty string if the key is not found
    }

    std::string getSettingsString() const { return settingsString_; }

    bool contains(const std::string& key) const { return settingsMap_.find(key) != settingsMap_.end(); }

    bool isSet() const
    {
        if (get("name") == "") return false;
        return true;
    }

    void remove(const std::string& key) { settingsMap_.erase(key); }

    void print() const
    {
        for (const auto& pair : settingsMap_)
        {
            std::cout << "Key: " << pair.first << ", Value: " << pair.second << std::endl;
        }
    }
};

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
    // TODO: need a better API for setting up compressor
    CompressorSettings                      cs;
    std::map<std::string, adios2::Operator> operators;

    // Rank-specific settings
    size_t numLocalParticles  = 0;
    size_t numGlobalParticles = 0;
    size_t numTotalRanks      = 0;
    size_t offset             = 0;
    size_t rank               = 0;
    size_t currStep           = 0; // Step, not numIteration

    ADIOS2Settings(const std::string& n)
        : cs(n)
    {
    }

    void setCompressor(const std::string& n) { cs = CompressorSettings(n); }

    std::string getCompressorSetting(const std::string& key, const std::string defaultValue = "")
    {
        return cs.get(key, defaultValue);
    }
};

// One executable should have 1 adios instance only
void initADIOSWriter(ADIOS2Settings& as)
{
#if ADIOS2_USE_MPI
    as.adios = adios2::ADIOS(as.comm);
#else
    as.adios = adios2::ADIOS();
#endif
    // TODO: will add this back very soon after the new release of ADIOS2
    // So that we can get rid of all the ifdefs

    // const char* const* list_operators = nullptr;
    // size_t             noperators     = 0;
    // adios2_available_operators(&noperators, &list_operators);
    as.io = as.adios.DeclareIO("bpio");
    as.io.DefineAttribute<std::string>("CompressorSettings", as.cs.getSettingsString(), "", "", true);
    if (as.cs.isSet())
    {
        if (as.getCompressorSetting("name") == "zfp")
        {
#ifdef ADIOS2_HAVE_ZFP
            as.operators["zfp"] = as.adios.DefineOperator("CompressorZFP", adios2::ops::LossyZFP);
#else
            throw std::runtime_error("Unsupported compressor ZFP. Compile ADIOS2 with ZFP to enable it.");
#endif
        }
        else if (as.getCompressorSetting("name") == "mgard")
        {
#ifdef ADIOS2_HAVE_MGARD
            as.operators["mgard"] = as.adios.DefineOperator("CompressorMGARD", adios2::ops::LossyMGARD);
#else
            throw std::runtime_error("Unsupported compressor MGARD. Compile ADIOS2 with MGARD to enable it.");
#endif
        }
        else if (as.getCompressorSetting("name") == "bzip2")
        {
#ifdef ADIOS2_HAVE_BZIP2
#else
            throw std::runtime_error("Unsupported compressor BZip2. Compile ADIOS2 with BZip2 to enable it.");
#endif
        }
        else if (as.getCompressorSetting("name") == "sz")
        {
#ifdef ADIOS2_HAVE_SZ
#else
            throw std::runtime_error("Unsupported compressor SZ. Compile ADIOS2 with SZ to enable it.");
#endif
        }
        else if (as.getCompressorSetting("name") == "png")
        {
#ifdef ADIOS2_HAVE_PNG
#else
            throw std::runtime_error("Unsupported compressor PNG(GZip). Compile ADIOS2 with PNG(GZip) to enable it.");
#endif
        }
    }
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
    adios2::Variable<T> var = as.io.DefineVariable<T>(fieldName,               // Field name
                                                      {as.numGlobalParticles}, // Global dimensions
                                                      {as.offset},             // Starting local offset
                                                      {as.numLocalParticles},  // Local dimensions (limited to 1 rank)
                                                      adios2::ConstantDims);
    if (as.cs.isSet())
    {
        if (as.getCompressorSetting("name") == "sz")
        {
#ifdef ADIOS2_HAVE_SZ
            var.AddOperation("sz", {{"accuracy", as.getCompressorSetting("accuracy", "0.0000001")}});
#else
            throw std::runtime_error("Unsupported compressor ZFP. Compile ADIOS2 with ZFP to enable it.");
#endif
        }
        else if (as.getCompressorSetting("name") == "zfp")
        {
#ifdef ADIOS2_HAVE_ZFP
            if (as.cs.contains("accuracy"))
            {
                var.AddOperation(as.operators["zfp"],
                                 {{adios2::ops::zfp::key::accuracy, as.getCompressorSetting("accuracy", "0.0000001")},
                                  {adios2::ops::zfp::key::backend, as.getCompressorSetting("backend", "omp")}});
            }
            else if (as.cs.contains("rate"))
            {

                var.AddOperation(as.operators["zfp"],
                                 {{adios2::ops::zfp::key::rate, as.getCompressorSetting("rate", "7")},
                                  {adios2::ops::zfp::key::backend, as.getCompressorSetting("backend", "omp")}});
            }
            else if (as.cs.contains("precision"))
            {
                var.AddOperation(as.operators["zfp"],
                                 {{adios2::ops::zfp::key::precision, as.getCompressorSetting("precision", "7")},
                                  {adios2::ops::zfp::key::backend, as.getCompressorSetting("backend", "omp")}});
            }
            else {}
#else
            throw std::runtime_error("Unsupported compressor ZFP. Compile ADIOS2 with ZFP to enable it.");
#endif
        }
        else if (as.getCompressorSetting("name") == "mgard")
        {
#ifdef ADIOS2_HAVE_MGARD
            if (as.cs.contains("accuracy"))
            {
                var.AddOperation(as.operators["mgard"], {{adios2::ops::mgard::key::accuracy,
                                                          as.getCompressorSetting("accuracy", "0.0000001")}});
            }
            else if (as.cs.contains("tolerance"))
            {
                var.AddOperation(as.operators["mgard"],
                                 {{adios2::ops::mgard::key::tolerance, as.getCompressorSetting("tolerance", "7")}});
            }
            else if (as.cs.contains("s"))
            {
                var.AddOperation(as.operators["mgard"],
                                 {{adios2::ops::mgard::key::s, as.getCompressorSetting("s", "0")}});
            }
            else {}
#else
            throw std::runtime_error("Unsupported compressor MGARD. Compile ADIOS2 with MGARD to enable it.");
#endif
        }
        else if (as.getCompressorSetting("name") == "bzip2")
        {
#ifdef ADIOS2_HAVE_BZIP2
            var.AddOperation(adios2::ops::LosslessBZIP2,
                             {{adios2::ops::bzip2::key::blockSize100k,
                               as.getCompressorSetting("blockSize100k", adios2::ops::bzip2::value::blockSize100k_9)}});
#else
            throw std::runtime_error("Unsupported compressor BZIP2. Compile ADIOS2 with BZIP2 to enable it.");
#endif
        }
        else if (as.getCompressorSetting("name") == "blosclz" || as.getCompressorSetting("name") == "lz4" ||
                 as.getCompressorSetting("name") == "lz4hc" || as.getCompressorSetting("name") == "zlib" ||
                 as.getCompressorSetting("name") == "zstd")
        {
#ifdef ADIOS2_HAVE_BLOSC2
            var.AddOperation(adios2::ops::LosslessBlosc,
                             {{adios2::ops::blosc::key::compressor, as.getCompressorSetting("name", "")},
                              {adios2::ops::blosc::key::clevel,
                               as.getCompressorSetting("level", adios2::ops::blosc::value::clevel_9)}});
#else
            throw std::runtime_error("Unsupported compressor. Compile ADIOS2 with BLOSC2 to enable it.");
#endif
        }
    }
    as.writer.Put(var, field);
}

template<class T>
void writeADIOSStepAttribute(ADIOS2Settings& as, const std::string& fieldName, const T* field, uint64_t size = 1)
{
    // StepAttribute is one variable with multiple copies (i.e. steps).
    // It's almost same as a field. Write only in rank 0.
    // There's no need for compression.
    if (as.rank == 0)
    {
        // StepAttribs are always double.
        if (size == 1)
        {
            adios2::Variable<T> var = as.io.DefineVariable<T>(fieldName);
            as.writer.Put(var, field);
        }
        else
        {
            adios2::Variable<T> var = as.io.DefineVariable<T>(fieldName, {size}, {0}, {size}, adios2::ConstantDims);
            as.writer.Put(var, field);
        }
    }
}

template<class T>
void writeADIOSFileAttribute(ADIOS2Settings& as, const std::string& fieldName, const T* field, uint64_t size = 1)
{
    // FileAttribute is a unique variable.
    // It is written into step 0. Write only by rank 0.
    // There's no need for compression.
    if (as.rank == 0)
    {
        // TODO: set them all to double
        if (size == 1) { as.io.DefineAttribute<T>(fieldName, (T)(field[0])); }
        else { as.io.DefineAttribute<T>(fieldName, field, size); }
    }
}

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

uint64_t ADIOSGetNumParticles(ADIOS2Settings& as)
{
    adios2::Variable<uint64_t> variable = as.io.InquireVariable<uint64_t>("numParticlesGlobal");
    variable.SetStepSelection({as.currStep - 1, 1});
    std::vector<uint64_t> varData(1);
    as.reader.Get(variable, varData.data(), adios2::Mode::Sync);
    return varData[0];
}

// This is step, not numIteration!!!
// Step is number of data groups within a checkpoint.
// Here the numSteps includes step0 (i.e. the step without actual particles)
// Returns the last Step where iteration is available
// Actual last step to read from should be {res.back() - 1, 1}
uint64_t ADIOSGetNumSteps(ADIOS2Settings& as)
{
    // iteration is a stepAttrib that gets written in all output formats
    adios2::Variable<uint64_t> variable = as.io.InquireVariable<uint64_t>("iteration");
    return variable.Steps();
}

auto ADIOSGetFileAttributes(ADIOS2Settings& as)
{
    std::map<std::string, adios2::Params> attribs = as.io.AvailableAttributes();
    // Skip "CompressorSettings" because it's not related to SPH
    attribs.erase("CompressorSettings");
    std::vector<std::string> res;
    std::ranges::transform(attribs, std::back_inserter(res), [](const auto& pair) { return pair.first; });
    return res;
}

uint64_t ADIOSGetFileAttributeSize(ADIOS2Settings& as, const std::string& key)
{
    std::map<std::string, adios2::Params> attribs = as.io.AvailableAttributes();
    // Skip "CompressorSettings" because it's not related to SPH
    attribs.erase("CompressorSettings");
    return std::stoi(attribs.at(key).at("Elements"));
}

uint64_t ADIOSGetStepAttributeSize(ADIOS2Settings& as, const std::string& key)
{
    const std::map<std::string, adios2::Params> attribs = as.io.AvailableAttributes();

    if (std::stoi(attribs.at(key).at("Elements")) > 1) { return std::stoi(attribs.at(key).at("Elements")); }
    else if (attribs.at(key).at("SingleValue") == "true") { return 1; }
    else { return std::stoi(attribs.at(key).at("Shape")); }
}

uint64_t ADIOSGetNumIterations(ADIOS2Settings& as)
{
    // iteration is a stepAttrib that gets written in all output formats, as long as it's not empty.
    if (as.currStep == 0)
    {
        throw std::runtime_error(
            "The imported file is empty! ADIOSGetNumIterations() should be called after ADIOSGetNumSteps().");
    }
    adios2::Variable<uint64_t> variable = as.io.InquireVariable<uint64_t>("iteration");
    variable.SetStepSelection({as.currStep - 1, 1});
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
void readADIOSStepAttribute(ADIOS2Settings& as, const std::string& fieldName, ExtractType* attr, uint64_t size = 1)
{
    if (as.currStep == 0)
    {
        throw std::runtime_error(
            "The imported file is empty! readADIOSStepAttribute() should be called after ADIOSGetNumSteps().");
    }
    // StepAttrib is an ADIOS single variable.
    // Can be read by every rank -- ADIOS will handle the MPI communication.
    if (size == 1)
    {
        adios2::Variable<ExtractType> variable = as.io.InquireVariable<ExtractType>(fieldName);
        variable.SetStepSelection({as.currStep - 1, 1});
        as.reader.Get(variable, attr, adios2::Mode::Sync);
    }
    else
    {
        adios2::Variable<ExtractType> variable = as.io.InquireVariable<ExtractType>(fieldName);
        variable.SetStepSelection({as.currStep - 1, 1});
        variable.SetSelection({{0}, {size}});
        as.reader.Get(variable, attr, adios2::Mode::Sync);
    }
}

std::string readADIOSFileCompressorSettings(ADIOS2Settings& as)
{

    auto interpretation_att = as.io.InquireAttribute<std::string>("CompressorSettings");
    if (interpretation_att)
    {
        std::string interpretation = interpretation_att.Data()[0];
        return interpretation;
    }
    return "";
}

template<class ExtractType>
void readADIOSFileAttribute(ADIOS2Settings& as, const std::string& fieldName, ExtractType* attr, uint64_t size = 1)
{
    // One MPI_COMM can only have one ADIOS instance.
    // Thus if we need another instance to write, has to recreate without the original as.comm.
    // File attribute is a real attribute.
    // Can be read by every rank -- ADIOS will handle the MPI communication.
    if (size == 1)
    {
        adios2::Attribute<ExtractType> res = as.io.InquireAttribute<ExtractType>(fieldName);
        attr[0]                            = res.Data()[0];
    }
    else
    {
        adios2::Attribute<ExtractType> res = as.io.InquireAttribute<ExtractType>(fieldName);
        for (u_int64_t i = 0; i < res.Data().size(); i++)
        {
            attr[i] = res.Data()[i];
        }
    }
}

} // namespace fileutils
} // namespace sphexa
