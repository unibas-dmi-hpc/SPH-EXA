#pragma once

#include <vector>

#include "kernels.hpp"

namespace sphexa
{
namespace sph
{

template<typename Dataset>
void computeEquationOfState(size_t startIndex, size_t endIndex, Dataset& d)
{
    using T = typename Dataset::RealType;

    const T R     = 8.317e7;
    const T gamma = (5.0 / 3.0);

    const T* ro  = d.ro.data();
    const T* mui = d.mui.data();
    const T* u   = d.u.data();

    T* temp = d.temp.data();
    T* p    = d.p.data();
    T* c    = d.c.data();
    T* cv   = d.cv.data();

#pragma omp parallel for schedule(static)
    for (size_t i = startIndex; i < endIndex; ++i)
    {
        cv[i]   = T(1.5) * R / mui[i];
        temp[i] = u[i] / cv[i];
        T tmp   = u[i] * (gamma - 1);
        p[i]    = ro[i] * tmp;
        c[i]    = std::sqrt(tmp);

#ifndef NDEBUG
        if (std::isnan(c[i]) || std::isnan(cv[i]))
            printf("ERROR:equation_of_state c %f cv %f temp %f u %f p %f\n", c[i], cv[i], temp[i], u[i], p[i]);
#endif
    }
}

} // namespace sph
} // namespace sphexa
