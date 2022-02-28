#pragma once

#include "cstone/findneighbors.hpp"

#include "task.hpp"

namespace sphexa
{

namespace sph
{

#ifndef USE_CUDA

template<class T, class KeyType>
void findNeighborsSfc(std::vector<Task>& taskList, gsl::span<const T> x, gsl::span<const T> y, gsl::span<const T> z,
                      gsl::span<const T> h, gsl::span<const KeyType> particleKeys, gsl::span<int> neighborsCount,
                      const cstone::Box<T>& box)
{
    std::array<std::size_t, 5> sizes{x.size(), y.size(), z.size(), h.size(), particleKeys.size()};
    if (std::count(begin(sizes), end(sizes), x.size()) != 5)
        throw std::runtime_error("findNeighborsSfc: input array sizes inconsistent\n");

    int ngmax = taskList.empty() ? 0 : taskList.front().ngmax;

    for (auto& t : taskList)
    {
        int* neighbors = t.neighbors.data();
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
                              neighborsCount.data() + t.firstParticle,
                              ngmax);
    }
}
#else

template<class T, class KeyType>
void findNeighborsSfc([[maybe_unused]] std::vector<Task>& taskList, [[maybe_unused]] gsl::span<const T> x,
                      [[maybe_unused]] gsl::span<const T> y, [[maybe_unused]] gsl::span<const T> z,
                      [[maybe_unused]] gsl::span<const T> h, [[maybe_unused]] const gsl::span<const KeyType> codes,
                      [[maybe_unused]] gsl::span<int> neighborsCount, [[maybe_unused]] const cstone::Box<T>& box)
{
}

#endif

size_t neighborsSum(size_t startIndex, size_t endIndex, gsl::span<const int> neighborsCount)
{
    size_t sum = 0;
#pragma omp parallel for reduction(+ : sum)
    for (size_t i = startIndex; i < endIndex; i++)
    {
        sum += neighborsCount[i];
    }

    int    rootRank  = 0;
    size_t globalSum = 0;
#ifdef USE_MPI
    MPI_Reduce(&sum, &globalSum, 1, MpiType<size_t>{}, MPI_SUM, rootRank, MPI_COMM_WORLD);
#endif

    return globalSum;
}

} // namespace sph
} // namespace sphexa
