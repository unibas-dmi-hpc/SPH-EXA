#pragma once

#include "cstone/findneighbors.hpp"

namespace sph
{

#ifndef USE_CUDA

template<class T, class KeyType>
void findNeighborsSfc(size_t startIndex, size_t endIndex, size_t ngmax, gsl::span<const T> x, gsl::span<const T> y,
                      gsl::span<const T> z, gsl::span<const T> h, gsl::span<const KeyType> particleKeys,
                      gsl::span<int> neighbors, gsl::span<int> neighborsCount, const cstone::Box<T>& box)
{
    std::array<std::size_t, 5> sizes{x.size(), y.size(), z.size(), h.size(), particleKeys.size()};
    if (std::count(begin(sizes), end(sizes), x.size()) != 5)
        throw std::runtime_error("findNeighborsSfc: input array sizes inconsistent\n");

    cstone::findNeighbors(x.data(),
                          y.data(),
                          z.data(),
                          h.data(),
                          startIndex,
                          endIndex,
                          x.size(),
                          box,
                          cstone::sfcKindPointer(particleKeys.data()),
                          neighbors.data(),
                          neighborsCount.data() + startIndex,
                          ngmax);
}
#else

template<class T, class KeyType>
void findNeighborsSfc(size_t, size_t, size_t, gsl::span<const T>, gsl::span<const T>, gsl::span<const T>,
                      gsl::span<const T>, gsl::span<const KeyType>, gsl::span<int>, gsl::span<int>,
                      const cstone::Box<T>&)
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
