#pragma once

#include <vector>
#include "Task.hpp"

namespace sphexa
{
namespace sph
{

template <typename T, class Dataset>
void updateSmoothingLengthImpl(Task &t, Dataset &d)
{
    const T c0 = 7.0;
    const T exp = 1.0 / 3.0;

    const int ng0 = t.ng0;
    const int *neighborsCount = t.neighborsCount.data();
    T *h = d.h.data();

    // general VE
    const T *m = d.m.data();
    T *xa = d.xa.data();
    const T *ro = d.ro.data();
    T *ballmass = d.ballmass.data();

    size_t n = t.clist.size();

#pragma omp parallel for schedule(guided)
    for (size_t pi = 0; pi < n; pi++)
    {
        int i = t.clist[pi];
        const int nn = neighborsCount[pi];

#ifdef DO_NEWTONRAPHSON
        if (d.iteration <= d.starthNR) { // only update it here if we are not doing NR yet. else it's done in the main loop for NR
            h[i] = h[i] * 0.5 * pow((1.0 + c0 * ng0 / nn), exp); // update of smoothing length...
            ballmass[i] = ro[i] * h[i] * h[i] * h[i];
        }
#else
        h[i] = h[i] * 0.5 * pow((1.0 + c0 * ng0 / nn), exp); // update of smoothing length...
#endif
        // also update VE estimator
#ifdef SPHYNX_VE
        if (d.iteration >= d.starthNR) {
            xa[i] = pow(m[i] / ro[i], d.veExp);  // sphynx VE...
        }
        else {
            xa[i] = m[i];  // "normal VE"
        }
#else
        xa[i] = m[i];  // "normal VE"
#endif

#ifndef NDEBUG
        if (std::isinf(h[i]) || std::isnan(h[i])) printf("ERROR::h(%d) ngi %d h %f\n", i, nn, h[i]);
#endif
    }
}

template <typename T, class Dataset>
void updateSmoothingLength(std::vector<Task> &taskList, Dataset &d)
{
    for (auto &task : taskList)
    {
        updateSmoothingLengthImpl<T>(task, d);
    }
}

} // namespace sph
} // namespace sphexa
