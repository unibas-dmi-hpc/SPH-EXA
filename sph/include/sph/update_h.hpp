#pragma once

#include <vector>

namespace sph
{

template<class Dataset>
void updateSmoothingLength(size_t startIndex, size_t endIndex, Dataset& d, unsigned ng0)
{
    using T     = typename Dataset::RealType;
    const T c0  = 7.0;
    const T exp = 1.0 / 3.0;

    const unsigned* neighborsCount = d.nc.data();
    T*              h              = d.h.data();

#pragma omp parallel for schedule(static)
    for (size_t i = startIndex; i < endIndex; i++)
    {
        unsigned nn = neighborsCount[i];
        h[i]        = h[i] * 0.5 * pow((1.0 + c0 * ng0 / nn), exp);

#ifndef NDEBUG
        if (std::isinf(h[i]) || std::isnan(h[i])) printf("ERROR::h(%lu) ngi %d h %f\n", i, nn, h[i]);
#endif
    }
}

} // namespace sph
