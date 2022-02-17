#pragma once

#include "cstone/findneighbors.hpp"

#include "task.hpp"

namespace sphexa
{

namespace sph
{

#ifndef USE_CUDA

template<class T, class KeyType>
void findNeighborsSfc(std::vector<Task>& taskList, const std::vector<T>& x, const std::vector<T>& y,
                      const std::vector<T>& z, const std::vector<T>& h, const std::vector<KeyType>& particleKeys,
                      const cstone::Box<T>& box)
{
    std::array<std::size_t, 5> sizes{x.size(), y.size(), z.size(), h.size(), particleKeys.size()};
    if (std::count(begin(sizes), end(sizes), x.size()) != 5)
        throw std::runtime_error("findNeighborsSfc: input array sizes inconsistent\n");

    int ngmax = taskList.empty() ? 0 : taskList.front().ngmax;

    for (auto& t : taskList)
    {
        int* neighbors      = t.neighbors.data();
        int* neighborsCount = t.neighborsCount.data();

        cstone::findNeighbors(x.data(),
                              y.data(),
                              z.data(),
                              h.data(),
                              t.firstParticle,
                              t.lastParticle,
                              x.size(),
                              box,
                              cstone::sfcKindPointer(particleKeys.data()),
                              neighbors,
                              neighborsCount,
                              ngmax);
    }
}
#else

template<class T, class KeyType>
void findNeighborsSfc([[maybe_unused]] std::vector<Task>& taskList, [[maybe_unused]] const std::vector<T>& x,
                      [[maybe_unused]] const std::vector<T>& y, [[maybe_unused]] const std::vector<T>& z,
                      [[maybe_unused]] const std::vector<T>& h, [[maybe_unused]] const std::vector<KeyType>& codes,
                      [[maybe_unused]] const cstone::Box<T>& box)
{
}

#endif

size_t neighborsSumImpl(const Task& t)
{
    size_t sum = 0;

#pragma omp parallel for reduction(+ : sum)
    for (size_t i = 0; i < t.size(); i++)
    {
        sum += t.neighborsCount[i];
    }

    return sum;
}

size_t neighborsSum(const std::vector<Task>& taskList)
{
    size_t sum = 0;

    for (const auto& task : taskList)
    {
        sum += neighborsSumImpl(task);
    }

#ifdef USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
#endif

    return sum;
}

} // namespace sph
} // namespace sphexa
