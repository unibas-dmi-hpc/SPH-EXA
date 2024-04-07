/*! @file
 * @brief Target particle group configuration
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/sfc/box.hpp"
#include "sph/sph_gpu.hpp"

namespace sph
{

//! @brief Compute spatial (=SFC-consecutive) groups of particles with compact bounding boxes
template<typename Tc, class Dataset>
void computeGroups(size_t startIndex, size_t endIndex, Dataset& d, const cstone::Box<Tc>& box,
                   GroupData<typename Dataset::AcceleratorType>& groups)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        computeSpatialGroups(startIndex, endIndex, d, box, groups);
    }
    else
    {
        groups.firstBody = startIndex;
        groups.lastBody  = endIndex;
    }
}

} // namespace sph
