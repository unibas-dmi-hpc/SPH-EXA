#pragma once

#include <vector>
#ifdef USE_CUDA
#include "pinned_allocator.h"
#endif

namespace sphexa
{

struct Task
{
    Task(const size_t ngmax, const size_t ng0)
        : ngmax(ngmax)
        , ng0(ng0)
        , firstParticle(0)
        , lastParticle(ngmax-1)
    {
    }

    void resize(const size_t size)
    {
#ifndef USE_CUDA
        neighbors.resize(size * ngmax);
#endif
        neighborsCount.resize(size);
    }

    const size_t ngmax;
    const size_t ng0;

    //! @brief first particle owned by rank, everything below is halos
    size_t firstParticle;
    //! @brief last particle owned by rank, everything above is halos
    size_t lastParticle;

    size_t size() const { return lastParticle - firstParticle; }

    std::vector<int> neighbors;
#ifdef USE_CUDA
    std::vector<int, pinned_allocator<int>> neighborsCount;
#else
    std::vector<int> neighborsCount;
#endif
};

class TaskList
{
public:
    TaskList(int firstIndex, int lastIndex, size_t nTasks, size_t ngmax, size_t ng0)
        : ngmax(ngmax)
        , ng0(ng0)
        , nTasks(nTasks)
        , tasks(nTasks, Task(ngmax, ng0))
    {
        update(firstIndex, lastIndex);
    }

    void update(int firstIndex, int lastIndex)
    {
        int numParticles = lastIndex - firstIndex;
        int partitionSize = numParticles / nTasks;
        int remainder = numParticles % nTasks;

        for (size_t i = 0; i < nTasks; ++i)
        {
            tasks[i].firstParticle = firstIndex + i * partitionSize;
            tasks[i].lastParticle  = firstIndex + (i + 1) * partitionSize + (i == nTasks - 1 ? remainder : 0);
            tasks[i].resize(tasks[i].size());
        }
    }

    const size_t ngmax;
    const size_t ng0;
    const size_t nTasks;
    std::vector<Task> tasks;

};
} // namespace sphexa
