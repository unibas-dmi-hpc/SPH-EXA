#pragma once

#include <cmath>

#include "cstone/cuda/cuda_utils.hpp"
#include "sph/sph_gpu.hpp"

namespace sph
{

template<class T>
void updateSmoothingLengthCpu(size_t startIndex, size_t endIndex, unsigned ng0, const unsigned* nc, T* h)
{
    // Note: these constants are duplicated in the GPU version, so don't forget to change them there as well
    constexpr double c0  = 7.0;
    constexpr double exp = 1.0 / 3.0;

#pragma omp parallel for schedule(static)
    for (size_t i = startIndex; i < endIndex; i++)
    {
        h[i] = h[i] * 0.5 * std::pow((1.0 + c0 * ng0 / nc[i]), exp);

#ifndef NDEBUG
        if (std::isinf(h[i]) || std::isnan(h[i])) printf("ERROR::h(%lu) ngi %d h %f\n", i, nc[i], h[i]);
#endif
    }
}

template<class Dataset>
void updateSmoothingLength(size_t startIndex, size_t endIndex, Dataset& d, unsigned ng0)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        updateSmoothingLengthGpu(startIndex, endIndex, ng0, rawPtr(d.devData.nc), rawPtr(d.devData.h));
    }
    else { updateSmoothingLengthCpu(startIndex, endIndex, ng0, rawPtr(d.nc), rawPtr(d.h)); }
}

} // namespace sph
