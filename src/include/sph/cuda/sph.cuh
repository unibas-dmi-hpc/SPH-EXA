#pragma once

#include <vector>

#include "SqPatch.hpp"

namespace sphexa
{
namespace sph
{
namespace cuda
{
using ParticleIdxChunk = std::vector<int>;

template <typename T, class Dataset>
extern void computeMomentumAndEnergy(const std::vector<ParticleIdxChunk> &l, Dataset &dataset);

template <typename T, class Dataset>
extern void computeDensity(const std::vector<ParticleIdxChunk> &l, Dataset &dataset);
}
} // namespace sph
} // namespace sphexa
