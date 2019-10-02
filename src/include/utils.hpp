#pragma once

#include <vector>
#include <algorithm>

namespace sphexa
{
namespace utils
{
template <typename T>
std::vector<std::vector<T>> partition(const std::vector<T> &l, const size_t numPartitions)
{
    const int partitionSize = l.size() / numPartitions;
    const int lastPartitionOffset = l.size() - numPartitions * partitionSize;

    auto partitioned = std::vector<std::vector<T>>(numPartitions);

    for (size_t i = 0; i < numPartitions; ++i)
    {
        const int begin = i * partitionSize;
        const int end = (i + 1) * partitionSize + (i == numPartitions - 1 ? lastPartitionOffset : 0);
        partitioned[i].reserve(end - begin);
        std::copy(l.begin() + begin, l.begin() + end, std::back_inserter(partitioned[i]));
    }

    return partitioned;
}
} // namespace utils
} // namespace sphexa
