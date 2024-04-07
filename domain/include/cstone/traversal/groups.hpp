/*! @file
 * @brief Data structures for particle target grouping
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/cuda/cuda_utils.hpp"
#include "cstone/tree/accel_switch.hpp"
#include "cstone/tree/definitions.h"

namespace cstone
{

//! @brief A view object to describe groups of target particles
struct GroupView
{
    LocalIndex firstBody, lastBody;
    LocalIndex numGroups;
    const LocalIndex* groupStart;
    const LocalIndex* groupEnd;
};

//! @brief Describes groups of spatially close particles that can be traversed through octrees in groups
template<class Accelerator>
class GroupData
{
    template<class T>
    using AccVector = typename AccelSwitchType<Accelerator, std::vector, thrust::device_vector>::template type<T>;

public:
    GroupView view() const { return {firstBody, lastBody, numGroups, groupStart, groupEnd}; }

    AccVector<LocalIndex> data;
    LocalIndex firstBody, lastBody;
    LocalIndex numGroups;
    LocalIndex* groupStart;
    LocalIndex* groupEnd;
};

} // namespace cstone
