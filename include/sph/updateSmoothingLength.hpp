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
        ballmass[i] = ro[i] * h[i] * h[i] * h[i]; //this is also in the findneighbors of sphynx (but only for those that need h adjusted...)

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
void updateSmoothingLengthForExceedingImpl(Task &t, Dataset &d, const size_t ngmin)
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
        const size_t nn = std::round(nn_actual[i]);

        // sphynx 1.5.3 has different updates if too many or too few than in normal case:
        //    if(nvi(i).gt.int(nvmax)) then  !note: if I saw it correctly at this point in the code nvi is either gt max or lt min...
        //       h(i)=max(0.9d0*h(i),h(i)*.5d0*(1.d0+hfac*0.9d0*nvmax/dble(nvi(i)))**hexp)
        //    else
        //       h(i)=h(i)*.5d0*(1.d0+hfac*1.1d0*nvmin/dble(nvi(i)))**hexp
        //    endif
        if (nn > t.ngmax || nn < ngmin) {
            if (nn > t.ngmax) {
                h[i] = std::max(0.9 * h[i], h[i] * 0.5 * pow((1.0 + c0 * 0.9 * t.ngmax / nn), exp)); // update of smoothing length if too many
            } else if (nn < ngmin) {
                h[i] = h[i] * 0.5 * pow((1.0 + c0 * 1.1 * ngmin / nn), exp); // update of smoothing length if too few
            }
            ballmass[i] = ro[i] * h[i] * h[i] * h[i];

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
void updateSmoothingLengthForExceeding(std::vector<Task> &taskList, Dataset &d, const size_t ngmin)
{
    for (auto &task : taskList)
    {
        updateSmoothingLengthForExceedingImpl<T>(task, d, ngmin);
    }
}
} // namespace sph
} // namespace sphexa
