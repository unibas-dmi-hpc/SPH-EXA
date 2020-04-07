//
// Created by Gabriel Zihlmann on 07.04.20.
//

#ifndef SPH_EXA_MINI_APP_UPDATEVEESTIMATOR_HPP
#define SPH_EXA_MINI_APP_UPDATEVEESTIMATOR_HPP

#endif //SPH_EXA_MINI_APP_UPDATEVEESTIMATOR_HPP

#pragma once

#include <vector>
#include "Task.hpp"

namespace sphexa
{
namespace sph
{

template <typename T, class Dataset>
struct XmassStdVE
{
    T operator()(const int i, const Dataset &d)
    {
        return d.m[i];
    }
};

template <typename T, class Dataset>
struct XmassSPHYNXVE
{
    T operator()(const int i, const Dataset &d)
    {
        return pow(d.m[i] / d.ro[i], d.veExp);
    }
};

template <typename T, class XmassFunct, class Dataset>
void updateVEEstimatorImpl(Task &t, Dataset &d)
{
    XmassFunct functXmass;
    T *xmass = d.xmass.data();

    size_t n = t.clist.size();

#pragma omp parallel for schedule(guided)
    for (size_t pi = 0; pi < n; pi++)
    {
        int i = t.clist[pi];
        // Update VE estimator
        xmass[i] = functXmass(i, d);

#ifndef NDEBUG
        if (std::isinf(xmass[i]) || std::isnan(xmass[i])) printf("ERROR::xmass(%d) xmass %f\n", int(d.id[i]), xmass[i]);
#endif
    }
}

template <typename T, class XmassFunct, class Dataset>
void updateVEEstimator(std::vector<Task> &taskList, Dataset &d)
{
    for (auto &task : taskList)
    {
        updateVEEstimatorImpl<T, XmassFunct>(task, d);
    }
}

} // namespace sph
} // namespace sphexa
