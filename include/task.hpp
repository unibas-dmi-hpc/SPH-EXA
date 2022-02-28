#pragma once

#include <vector>
#ifdef USE_CUDA
#include "pinned_allocator.h"
#endif

namespace sphexa
{

struct Task
{
    Task(size_t ngmax)
        : ngmax(ngmax)
    {
    }

    void resize(const size_t size)
    {
#ifndef USE_CUDA
        neighbors.resize(size * ngmax);
#endif
    }

    const size_t ngmax;

    //! @brief first particle owned by rank, everything below is halos
    size_t firstParticle{0};
    //! @brief last particle owned by rank, everything above is halos
    size_t lastParticle{0};

    size_t size() const { return lastParticle - firstParticle; }

    std::vector<int> neighbors;
};

class TaskList
{
public:
    TaskList(size_t nTasks, size_t ngmax)
        : ngmax(ngmax)
        , tasks(nTasks, Task(ngmax))
    {
    }

    void update(size_t firstIndex, size_t lastIndex)
    {
        size_t numTasks      = tasks.size();
        size_t numParticles  = lastIndex - firstIndex;
        size_t partitionSize = numParticles / numTasks;
        size_t remainder     = numParticles % numTasks;

        for (size_t i = 0; i < numTasks; ++i)
        {
            tasks[i].firstParticle = firstIndex + i * partitionSize;
            tasks[i].lastParticle  = firstIndex + (i + 1) * partitionSize + (i == numTasks - 1 ? remainder : 0);
            tasks[i].resize(tasks[i].size());
        }
    }

    const size_t      ngmax;
    std::vector<Task> tasks;
};
} // namespace sphexa
