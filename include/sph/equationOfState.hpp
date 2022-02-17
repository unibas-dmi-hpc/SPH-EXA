#pragma once

#include <vector>

#include "kernels.hpp"

namespace sphexa
{
namespace sph
{

template<typename T, typename Dataset>
void computeEquationOfStateEvrardImpl(const Task& t, Dataset& d)
{
    size_t numParticles = t.size();

    const T R = 8.317e7, gamma = (5.0 / 3.0);

    const T* ro  = d.ro.data();
    const T* mui = d.mui.data();
    const T* u   = d.u.data();

    T* temp = d.temp.data();
    T* p    = d.p.data();
    T* c    = d.c.data();
    T* cv   = d.cv.data();

#pragma omp parallel for
    for (size_t pi = 0; pi < numParticles; ++pi)
    {
        int i = pi + t.firstParticle;

        cv[i]   = 1.5 * R / mui[i];
        temp[i] = u[i] / cv[i];
        T tmp   = u[i] * (gamma - 1);
        p[i]    = ro[i] * tmp;
        c[i]    = sqrt(tmp);

        if (std::isnan(c[i]) || std::isnan(cv[i]))
            printf("ERROR:equation_of_state c %f cv %f temp %f u %f p %f\n", c[i], cv[i], temp[i], u[i], p[i]);
    }
}
template<typename T, class Dataset>
void computeEquationOfStateEvrard(const std::vector<Task>& taskList, Dataset& d)
{
    for (const auto& task : taskList)
    {
        computeEquationOfStateEvrardImpl<T>(task, d);
    }
}
} // namespace sph
} // namespace sphexa
