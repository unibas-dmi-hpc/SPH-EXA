/*! @file
 * @brief Data structures for particle target grouping
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <utility>

#include "cstone/cuda/device_vector.h"
#include "cstone/primitives/accel_switch.hpp"
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
    using AccVector = typename AccelSwitchType<Accelerator, std::vector, DeviceVector>::template type<T>;

public:
    GroupData()                 = default;
    GroupData(const GroupView&) = delete;
    GroupView view() const { return {firstBody, lastBody, numGroups, groupStart, groupEnd}; }

    AccVector<LocalIndex> data;
    LocalIndex firstBody, lastBody;
    LocalIndex numGroups;
    LocalIndex* groupStart;
    LocalIndex* groupEnd;

private:
    friend void swap(GroupData& lhs, GroupData& rhs)
    {
        swap(lhs.data, rhs.data);
        std::swap(lhs.firstBody, rhs.firstBody);
        std::swap(lhs.lastBody, rhs.lastBody);
        std::swap(lhs.numGroups, rhs.numGroups);
        std::swap(lhs.groupStart, rhs.groupStart);
        std::swap(lhs.groupEnd, rhs.groupEnd);
    }
};

} // namespace cstone
