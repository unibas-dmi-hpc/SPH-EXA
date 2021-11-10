#pragma once

#include <string>
#include <vector>

namespace sphexa
{
template <typename Dataset>
struct IFileWriter
{
#ifdef SPH_EXA_HAVE_H5PART
    virtual void dumpParticleDataToH5File(const Dataset& d, int firstIndex, int lastIndex, const std::string &path) const = 0;
#endif
    virtual void dumpParticleDataToBinFile(const Dataset& d, const std::string &path) const = 0;

    virtual void dumpParticleDataToAsciiFile(const Dataset& d, int firstIndex, int lastIndex,
                                             const std::string& path) const                   = 0;

    virtual void dumpCheckpointDataToBinFile(const Dataset& d, const std::string& path) const = 0;

    virtual ~IFileWriter() = default;
};
} // namespace sphexa
