//
// Created by Gabriel Zihlmann on 17.02.20.
//

#ifndef SPH_EXA_MINI_APP_NEWTONRAPHSON_HPP
#define SPH_EXA_MINI_APP_NEWTONRAPHSON_HPP

#endif //SPH_EXA_MINI_APP_NEWTONRAPHSON_HPP

#pragma once

#include <vector>

#include "../ParticlesData.hpp"
#include "kernels.hpp"
#include "Task.hpp"
#include "lookupTables.hpp"
#include "cuda/sph.cuh"

namespace sphexa
{
    namespace sph
    {

        template <typename T, class Dataset>
        void newtonRaphsonImpl(const Task &t, Dataset &d)
        {
            const size_t n = t.clist.size();
            const int *clist = t.clist.data();


            T *h = d.h.data();
            const T *ro = d.ro.data();
            // general VE
            const T *sumwh = d.sumwh.data(); // the sum of Xmass weighted by the derivative of the kernel wrt. h
            const T *ballmass = d.ballmass.data();

            T f, fprime, deltah;

#pragma omp parallel for private(f, fprime, deltah)
            for (size_t pi = 0; pi < n; pi++)
            {
                const int i = clist[pi];
                f = ballmass[i] / (h[i] * h[i] * h[i]) - ro[i];  // function we want to find the 0 of
                fprime = - 3 * ballmass[i] / (h[i] * h[i] * h[i] * h[i]) - sumwh[i];  // df/dh

                deltah = - f / fprime;

                if (abs(deltah / h[i]) < 0.2) {
                    h[i] += deltah;  // only update if smaller than 0.2...
                    // but what happens if deltah/h is >= 0.2? As h doesn't change, sumwh, sumkx and thus the density
                    // do not change in this entire timestep -> it won't get updated in any future NR iteration!
                }
                else {
//                    printf("h[%d] not updated (deltah/h was %.3f)\n", i, deltah / h[i]);
                }

#ifndef NDEBUG
                if (std::isnan(h[i])) printf("ERROR::NewtonRaphson(%d) h %f, delta-h: %f\n", i, h[i], deltah);
#endif
            }
        }

        template <typename T, class Dataset>
        void newtonRaphson(const std::vector<Task> &taskList, Dataset &d)
        {
            for (const auto &task : taskList)
            {
                newtonRaphsonImpl<T>(task, d);
            }
        }

    } // namespace sph
} // namespace sphexa
