#pragma once

#include <vector>
#include "Task.hpp"

namespace sphexa
{
namespace sph
{

template <typename T, class Dataset>
void findNeighborsImpl(const Octree<T> &o, Task &t, Dataset &d)
{
    const size_t n = t.clist.size();
    T *nn = d.nn.data();
    T *nn_actual = d.nn_actual.data();

#pragma omp parallel for schedule(guided)
    for (size_t pi = 0; pi < n; pi++)
    {
        const int i = t.clist[pi];

        t.neighborsCount[pi] = 0;
        int nn_act = 0;
        o.findNeighbors(i, &d.x[0], &d.y[0], &d.z[0], d.x[i], d.y[i], d.z[i], 2.0 * d.h[i], t.ngmax, &t.neighbors[pi * t.ngmax],
                             t.neighborsCount[pi], nn_act, d.bbox.PBCx, d.bbox.PBCy, d.bbox.PBCz);

#ifndef NDEBUG
        if (t.neighborsCount[pi] == 0)
            printf("ERROR::FindNeighbors(%d) x %f y %f z %f h = %f ngi %d\n", int(d.id[i]), d.x[i], d.y[i], d.z[i], d.h[i], t.neighborsCount[pi]);
        if (t.neighborsCount[pi] == t.ngmax)
            printf("WARNING::FindNeighbors(%d) x %f y %f z %f h = %f ngi %d reached ngmax (%d). Actual neighbor count is %d\n", int(d.id[i]), d.x[i], d.y[i], d.z[i], d.h[i], t.neighborsCount[pi], t.ngmax, nn_act);
#endif
        nn[i] = t.neighborsCount[pi];
        nn_actual[i] = nn_act;
    }
}

template <typename T, class Dataset>
void findNeighbors(const Octree<T> &o, std::vector<Task> &taskList, Dataset &d)
{
    for (auto &task : taskList)
    {
        findNeighborsImpl(o, task, d);
    }
}

size_t neighborsSumImpl(const Task &t)
{
    size_t sum = 0;
#pragma omp parallel for reduction(+ : sum)
    for (unsigned int i = 0; i < t.clist.size(); i++)
        sum += t.neighborsCount[i];
    return sum;
}

size_t neighborsSum(const std::vector<Task> &taskList)
{
    size_t sum = 0;
    for (const auto &task : taskList)
    {
        sum += neighborsSumImpl(task);
    }

#ifdef USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
#endif

    return sum;
}

template <typename T, class Dataset>
size_t neighborsMaxImpl(const Task &t, Dataset &d)
{
    size_t max = 0;
#pragma omp parallel for reduction(max : max)
    for (unsigned int pi = 0; pi < t.clist.size(); pi++){
        int i = t.clist[pi];
        max = d.nn_actual[i] > max ? d.nn_actual[i] : max;
    }
    return max;
}

template <typename T, class Dataset>
size_t neighborsMax(const std::vector<Task> &taskList, Dataset &d)
{
    size_t max = 0;
    for (const auto &task : taskList)
    {
        size_t m = neighborsMaxImpl<T>(task, d);
        max = m > max ? m : max;
    }

#ifdef USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, &max, 1, MPI_LONG_LONG_INT, MPI_MAX, MPI_COMM_WORLD);
#endif

    return max;
}
} // namespace sph
} // namespace sphexa
