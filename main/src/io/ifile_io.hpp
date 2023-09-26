/*
 * MIT License
 *
 * Copyright (c) 2023 CSCS, ETH Zurich, University of Basel, University of Zurich
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
 * @brief file I/O interface
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <map>
#include <string>
#include <variant>

#include "cstone/sfc/box.hpp"
#include "cstone/util/type_list.hpp"

namespace sphexa
{

struct IO
{
    template<class T>
    using ConstPtr = const T*;

    using Types = util::TypeList<double, float, char, int, int64_t, unsigned, uint64_t>;
};

class IFileWriter
{
    template<class T>
    using ToVec = const std::vector<T>;

public:
    using FieldType   = util::Reduce<std::variant, util::Map<IO::ConstPtr, IO::Types>>;
    using FieldVector = util::Reduce<std::variant, util::Map<ToVec, IO::Types>>;

    virtual void constants(const std::map<std::string, double>& c, std::string path) const = 0;

    virtual std::string suffix() const = 0;

    virtual ~IFileWriter() = default;

    virtual void    addStep(size_t firstIndex, size_t lastIndex, std::string path) = 0;
    virtual int64_t stepAttributeSize(const std::string& /*key*/) { return 0; }
    virtual void    stepAttribute(const std::string& key, FieldType val, int64_t size) = 0;
    virtual void    writeField(const std::string& key, FieldType field, int col)       = 0;
    virtual void    setNumParticles(uint64_t numParticles)                             = 0;
    virtual void    setCompression(const std::string & compressionMethod, int compressionParam)                          = 0;
    virtual void    closeStep()                                                        = 0;
};

class IFileReader
{
public:
    using FieldType = util::Reduce<std::variant, util::Map<std::add_pointer_t, IO::Types>>;

    virtual ~IFileReader() = default;

    virtual void                     setStep(std::string path, int step)                                = 0;
    virtual std::vector<std::string> fileAttributes()                                                   = 0;
    virtual std::vector<std::string> stepAttributes()                                                   = 0;
    virtual int64_t                  fileAttributeSize(const std::string& key)                          = 0;
    virtual int64_t                  stepAttributeSize(const std::string& key)                          = 0;
    virtual void                     fileAttribute(const std::string& key, FieldType val, int64_t size) = 0;
    virtual void                     stepAttribute(const std::string& key, FieldType val, int64_t size) = 0;
    virtual void                     readField(const std::string& key, FieldType field)                 = 0;
    virtual uint64_t                 localNumParticles()                                                = 0;
    virtual uint64_t                 globalNumParticles()                                               = 0;
    virtual void                     closeStep()                                                        = 0;
};

//! @brief Used to initialize particle dataset attributes from builtin named test-cases
class BuiltinWriter
{
    // value_type should be generalized to std::variant
    using AttributeMap = std::map<std::string, double>;

public:
    using FieldType = util::Reduce<std::variant, util::Map<std::add_pointer_t, IO::Types>>;

    explicit BuiltinWriter(AttributeMap attrs)
        : attributes_(std::move(attrs))
    {
    }

    void stepAttribute(const std::string& key, FieldType val, int64_t /*size*/)
    {
        std::visit([this, &key](auto arg) { *arg = attributes_.at(key); }, val);
    };

private:
    AttributeMap attributes_;
};

//! @brief Used to read the default values of dataset attributes
class BuiltinReader
{
    // value_type should be generalized to std::variant
    using AttributeMap = std::map<std::string, double>;

public:
    using FieldType = util::Reduce<std::variant, util::Map<std::add_pointer_t, IO::Types>>;

    explicit BuiltinReader(AttributeMap& attrs)
        : attributes_(attrs)
    {
    }

    void stepAttribute(const std::string& key, FieldType val, int64_t /*size*/)
    {
        std::visit([this, &key](auto arg) { attributes_[key] = *arg; }, val);
    };

private:
    //! @brief reference to attributes
    AttributeMap& attributes_;
};

} // namespace sphexa
