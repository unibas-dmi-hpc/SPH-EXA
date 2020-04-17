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
    const T *nn_actual = d.nn_actual.data();
    T *h = d.h.data();

    const T *ro = d.ro.data();
    T *ballmass = d.ballmass.data();

    size_t n = t.clist.size();

#pragma omp parallel for schedule(guided)
    for (size_t pi = 0; pi < n; pi++)
    {
        int i = t.clist[pi];
//        const int nn = neighborsCount[pi];
        const int nn = std::round(nn_actual[i]);

        h[i] = h[i] * 0.5 * pow((1.0 + c0 * ng0 / nn), exp); // update of smoothing length...
        ballmass[i] = ro[i] * h[i] * h[i] * h[i]; //this is also in the findneighbors of sphynx -> runs every iteration, not just if iter > startNR as it is in update of sphynx

#ifndef NDEBUG
        if (std::isinf(h[i]) || std::isnan(h[i])) printf("ERROR::h(%d) ngi %d h %f\n", int(d.id[i]), nn, h[i]);
        if ((d.bbox.PBCx && 2.0 * h[i] > (d.bbox.xmax - d.bbox.xmin) / 2.0) ||
            (d.bbox.PBCy && 2.0 * h[i] > (d.bbox.ymax - d.bbox.ymin) / 2.0) ||
            (d.bbox.PBCz && 2.0 * h[i] > (d.bbox.zmax - d.bbox.zmin) / 2.0)
                )
            printf("ERROR::Update_h(%d) 2*(h=%f) > than half of domain width! x: %f - %f, y: %f - %f, z: %f - %f\n",
                   int(d.id[i]), h[i], d.bbox.xmin, d.bbox.xmax, d.bbox.ymin, d.bbox.ymax, d.bbox.zmin, d.bbox.zmax);
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

template <typename T, class Dataset>
void updateSmoothingLengthForExceedingImpl(Task &t, Dataset &d)
{
    const T c0 = 7.0;
    const T exp = 1.0 / 3.0;

    const int ng0 = t.ng0;
    const T *nn_actual = d.nn_actual.data();
    T *h = d.h.data();

    const T *ro = d.ro.data();
    T *ballmass = d.ballmass.data();

    size_t n = t.clist.size();

#pragma omp parallel for schedule(guided)
    for (size_t pi = 0; pi < n; pi++)
    {
        int i = t.clist[pi];
//        const int nn = neighborsCount[pi];
        const int nn = std::round(nn_actual[i]);

        if (nn > t.ngmax){
            h[i] = h[i] * 0.5 * pow((1.0 + c0 * ng0 / nn), exp); // update of smoothing length...
            ballmass[i] = ro[i] * h[i] * h[i] * h[i]; //this is also in the findneighbors of sphynx -> runs every iteration, not just if iter > startNR as it is in update of sphynx

#ifndef NDEBUG
            if (std::isinf(h[i]) || std::isnan(h[i])) printf("ERROR::h(%d) ngi %d h %f\n", int(d.id[i]), nn, h[i]);
            if ((d.bbox.PBCx && 2.0 * h[i] > (d.bbox.xmax - d.bbox.xmin) / 2.0) ||
                (d.bbox.PBCy && 2.0 * h[i] > (d.bbox.ymax - d.bbox.ymin) / 2.0) ||
                (d.bbox.PBCz && 2.0 * h[i] > (d.bbox.zmax - d.bbox.zmin) / 2.0)
                    )
                printf("ERROR::Update_h(%d) 2*(h=%f) > than half of domain width! x: %f - %f, y: %f - %f, z: %f - %f\n",
                       int(d.id[i]), h[i], d.bbox.xmin, d.bbox.xmax, d.bbox.ymin, d.bbox.ymax, d.bbox.zmin, d.bbox.zmax);
#endif
        }
    }
}

template <typename T, class Dataset>
void updateSmoothingLengthForExceeding(std::vector<Task> &taskList, Dataset &d)
{
    for (auto &task : taskList)
    {
        updateSmoothingLengthForExceedingImpl<T>(task, d);
    }
}
} // namespace sph
} // namespace sphexa
