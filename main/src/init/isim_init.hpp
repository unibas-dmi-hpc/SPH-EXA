/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief Test-case simulation data initialization
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <filesystem>
#include <map>

#include "cstone/sfc/box.hpp"
#include "io/ifile_io.hpp"

namespace sphexa
{

using InitSettings = std::map<std::string, double>;

template<class Dataset>
class ISimInitializer
{
public:
    virtual cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, size_t, Dataset& d,
                                                         IFileReader*) const = 0;

    virtual const InitSettings& constants() const = 0;

    virtual ~ISimInitializer() = default;
};

//! @brief Used to initialize particle dataset attributes from builtin named test-cases
class BuiltinWriter
{
public:
    using FieldType = util::Reduce<std::variant, util::Map<std::add_pointer_t, IO::Types>>;

    explicit BuiltinWriter(InitSettings attrs)
        : attributes_(std::move(attrs))
    {
    }

    [[nodiscard]] static int rank() { return -1; }

    void stepAttribute(const std::string& key, FieldType val, int64_t /*size*/)
    {
        std::visit([this, &key](auto arg) { *arg = attributes_.at(key); }, val);
    };

private:
    InitSettings attributes_;
};

//! @brief Used to read the default values of dataset attributes
class BuiltinReader
{
public:
    using FieldType = util::Reduce<std::variant, util::Map<std::add_pointer_t, IO::Types>>;

    explicit BuiltinReader(InitSettings& attrs)
        : attributes_(attrs)
    {
    }

    [[nodiscard]] static int rank() { return -1; }

    void stepAttribute(const std::string& key, FieldType val, int64_t /*size*/)
    {
        std::visit([this, &key](auto arg) { attributes_[key] = *arg; }, val);
    };

private:
    //! @brief reference to attributes
    InitSettings& attributes_;
};

void writeSettings(const InitSettings& settings, const std::string& path, IFileWriter* writer)
{
    if (std::filesystem::exists(path))
    {
        throw std::runtime_error("Cannot write settings: file " + path + " already exists\n");
    }

    writer->addStep(0, 0, path);
    for (auto it = settings.cbegin(); it != settings.cend(); ++it)
    {
        writer->fileAttribute(it->first, &(it->second), 1);
    }
    writer->closeStep();
}

//! @brief read file attributes into an associative container
void readFileAttributes(InitSettings& settings, const std::string& settingsFile, IFileReader* reader, bool verbose)
{
    if (not settingsFile.empty())
    {
        reader->setStep(settingsFile, -1, FileMode::independent);

        auto fileAttributes = reader->fileAttributes();
        for (const auto& attr : fileAttributes)
        {
            int64_t sz = reader->fileAttributeSize(attr);
            if (sz == 1)
            {
                settings[attr] = {};
                reader->fileAttribute(attr, &settings[attr], sz);
                if (reader->rank() == 0 && verbose)
                {
                    std::cout << "Override setting from " << settingsFile << ": " << attr << " = " << settings[attr]
                              << std::endl;
                }
            }
        }
        reader->closeStep();
    }
}

//! @brief build up an associative container with test case settings
template<class Dataset>
[[nodiscard]] InitSettings buildSettings(Dataset&& d, const InitSettings& testCaseSettings,
                                         const std::string& settingsFile, IFileReader* reader)
{
    InitSettings settings;
    // first layer: class member defaults in code
    BuiltinReader extractor(settings);
    d.hydro.loadOrStoreAttributes(&extractor);

    // second layer: test-case specific settings
    for (const auto& kv : testCaseSettings)
    {
        settings[kv.first] = kv.second;
    }

    // third layer: settings override by file given on commandline (highest precedence)
    readFileAttributes(settings, settingsFile, reader, true);

    return settings;
}

} // namespace sphexa