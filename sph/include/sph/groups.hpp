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

//! @brief Describes groups of spatially close particles that can be traversed through octrees in groups
template<class Accelerator>
class TargetGroups
{
    using AccVector = typename cstone::AccelSwitchType<Accelerator, std::vector,
                                                       thrust::device_vector>::template type<cstone::LocalIndex>;

public:
    cstone::LocalIndex numGroups() const { return groups_.size() - 1; }

private:
    AccVector groups_;
    cstone::LocalIndex firstBody_, lastBody_;
    const cstone::LocalIndex* grpStart_;
    const cstone::LocalIndex* grpEnd_;
};

template<typename Tc, class Dataset>
void computeGroups(size_t startIndex, size_t endIndex, Dataset& d, const cstone::Box<Tc>& box,
                   TargetGroups<typename Dataset::AcceleratorType>& groups)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        computeTargetGroups(startIndex, endIndex, d, box);
    }
}

} // namespace sph
