#pragma once

#include <vector>
#include <math.h>
#include <algorithm>

#include "kernels.hpp"

#ifdef USE_MPI
#include "mpi.h"
#endif

namespace sphexa
{
namespace sph
{

// TODO: rename into KCourant
template <typename T, class Dataset>
struct TimestepPress2ndOrder
{
    T operator()(const int i, const Dataset &d)
    {
        const T maxvsignal = d.maxvsignal[i];
        return maxvsignal > 0.0 ? d.Kcour * d.h[i] / d.maxvsignal[i] : d.Kcour * d.h[i] / d.c[i];
    }
};

template <typename T, class Dataset>
struct TimestepKCourantDmy
{
    T operator()(const int i, const Dataset &d)
    {
        const T dtu = std::abs(d.dku * (d.u[i] / d.du[i]));
        const T dtcour = d.Kcour * d.h[i] / d.c[i];
        return std::min(dtu, dtcour);
    }
};

template <typename T, class TimestepFunct, class Dataset>
void computeMinTimestepImpl(const Task &t, Dataset &d, T &minDt)
{
    TimestepFunct functTimestep;
    int numParticles = t.size();

    T minDtTmp = INFINITY;

    #pragma omp parallel for reduction(min : minDtTmp)
    for (size_t pi = 0; pi < numParticles; pi++)
    {
        int i = pi + t.firstParticle;
        const T dt_i = functTimestep(i, d);

        minDtTmp = std::min(minDtTmp, dt_i);
    }

    minDt = minDtTmp;
}

template <typename T, class Dataset>
void setMinTimestepImpl(const Task &t, Dataset &d, const T minDt)
{
    int numParticles = t.size();

    T* dt = d.dt.data();

    #pragma omp parallel for
    for (size_t pi = 0; pi < numParticles; pi++)
    {
        int i = pi + t.firstParticle;
        dt[i] = minDt;
    }
}

template <typename T, class TimestepFunct, class Dataset>
void computeTimestep(const std::vector<Task> &taskList, Dataset &d)
{
    T minDtTmp = INFINITY;
    for (const auto &task : taskList)
    {
        T tminDt = 0.0;
        computeMinTimestepImpl<T, TimestepFunct>(task, d, tminDt);
        minDtTmp = std::min(tminDt, minDtTmp);
    }

    //    d.minTmpDt = minDtTmp;
    minDtTmp = std::min(minDtTmp, d.maxDtIncrease * d.minDt);

#ifdef USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, &minDtTmp, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
#endif

    for (const auto &task : taskList)
    {
        setMinTimestepImpl(task, d, minDtTmp);
    }

    d.ttot += minDtTmp;
    d.minDt = minDtTmp;
}

} // namespace sph
} // namespace sphexa
