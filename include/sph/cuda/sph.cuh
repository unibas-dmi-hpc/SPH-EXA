#pragma once

#include <vector>

#include "Task.hpp"
#include "cudaParticlesData.cuh"
#include "cstone/sfc/box.hpp"

namespace sphexa
{
namespace sph
{
namespace cuda
{

template <class Dataset>
extern void computeDensity(std::vector<Task>& taskList, Dataset& d, const cstone::Box<double>&);

template <class Dataset>
extern void computeIAD(const std::vector<Task>& taskList, Dataset& d, const cstone::Box<double>&);

template <class Dataset>
extern void computeMomentumAndEnergyIAD(const std::vector<Task>& taskList, Dataset& d, const cstone::Box<double>&);

} // namespace cuda
} // namespace sph
} // namespace sphexa
