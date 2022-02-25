#pragma once

#include <vector>

#include "task.hpp"
#include "gpu_particle_data.cuh"
#include "cstone/sfc/box.hpp"

//! @brief maximum number of neighbors supported in GPU kernels
#define NGMAX 150

namespace sphexa
{
namespace sph
{
namespace cuda
{

template<class Dataset>
extern void computeDensity(std::vector<Task>& taskList, Dataset& d, const cstone::Box<double>&);

template<class Dataset>
extern void computeIAD(const std::vector<Task>& taskList, Dataset& d, const cstone::Box<double>&);

template<class Dataset>
extern void computeMomentumAndEnergyIAD(const std::vector<Task>& taskList, Dataset& d, const cstone::Box<double>&);

} // namespace cuda
} // namespace sph
} // namespace sphexa
