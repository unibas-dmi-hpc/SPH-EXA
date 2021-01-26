#pragma once

#include <vector>

#include "Task.hpp"
#if defined(USE_CUDA)
#include "cudaParticlesData.cuh"
#endif

namespace sphexa
{
namespace sph
{
namespace cuda
{

template <typename T, class Dataset>
extern void computeFindNeighbors(const LinearOctree<T> &o, std::vector<Task> &taskList, Dataset &d);

template <typename T, class Dataset>
extern void computeDensity(std::vector<Task> &taskList, Dataset &d);

template <typename T, class Dataset>
extern void computeMomentumAndEnergy(const std::vector<Task> &taskList, Dataset &d);

template <typename T, class Dataset>
extern void computeIAD(const std::vector<Task> &taskList, Dataset &d);

template <typename T, class Dataset>
extern void computeMomentumAndEnergyIAD(const std::vector<Task> &taskList, Dataset &d);

} // namespace cuda
} // namespace sph
} // namespace sphexa
