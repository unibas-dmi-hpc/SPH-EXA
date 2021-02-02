#pragma once

#include "cstone/findneighbors.hpp"

#include "Task.hpp"

namespace sphexa
{

namespace sph
{

#ifndef USE_CUDA

template<class T, class I>
void findNeighborsSfc(std::vector<Task>& taskList,
                      const std::vector<T>& x,
                      const std::vector<T>& y,
                      const std::vector<T>& z,
                      const std::vector<T>& h,
                      const std::vector<I>& codes,
                      const cstone::Box<T>& box)
{
    std::array<std::size_t, 5> sizes{x.size(), y.size(), z.size(), h.size(), codes.size()};
    if (std::count(begin(sizes), end(sizes), x.size()) != 5)
        throw std::runtime_error("findNeighborsSfc: input array sizes inconsistent\n");

    int ngmax = taskList.empty() ? 0 : taskList.front().ngmax;

    for (auto &t : taskList)
    {
        const int* clist = t.clist.data();
        int* neighbors = t.neighbors.data();
        int* neighborsCount = t.neighborsCount.data();

        const size_t n = t.clist.size();

        #pragma omp parallel for
        for (size_t pi = 0; pi < n; pi++)
        {
            int i = clist[pi];
            cstone::findNeighbors(i, x.data(), y.data(), z.data(), h.data(), box, codes.data(),
                                  neighbors + pi*ngmax, neighborsCount + pi, x.size(), ngmax);
        }
    }
}
#else

template<class T, class I>
void findNeighborsSfc(std::vector<Task>& taskList,
                      const std::vector<T>& x,
                      const std::vector<T>& y,
                      const std::vector<T>& z,
                      const std::vector<T>& h,
                      const std::vector<I>& codes,
                      const cstone::Box<T>& box)
{

}

#endif

} // namespace sph
} // namespace sphexa