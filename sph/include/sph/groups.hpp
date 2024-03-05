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

//! @brief A view object to describe groups of target particles
struct GroupView
{
    cstone::LocalIndex        firstBody, lastBody;
    cstone::LocalIndex        numGroups;
    const cstone::LocalIndex* groupStart;
    const cstone::LocalIndex* groupEnd;
};

//! @brief Describes groups of spatially close particles that can be traversed through octrees in groups
template<class Accelerator>
class GroupData
{
    using LocalIndex = cstone::LocalIndex;

    template<class T>
    using AccVector =
        typename cstone::AccelSwitchType<Accelerator, std::vector, thrust::device_vector>::template type<T>;

public:
    GroupView view() const { return {firstBody, lastBody, numGroups, groupStart, groupEnd}; }

    AccVector<LocalIndex> data;
    LocalIndex            firstBody, lastBody;
    LocalIndex            numGroups;
    LocalIndex*           groupStart;
    LocalIndex*           groupEnd;
};

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
