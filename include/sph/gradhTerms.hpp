//
// Created by Gabriel Zihlmann on 02.03.20.
//

#ifndef SPH_EXA_MINI_APP_GRADHTERMS_HPP
#define SPH_EXA_MINI_APP_GRADHTERMS_HPP

#endif //SPH_EXA_MINI_APP_GRADHTERMS_HPP

#pragma once

#include <vector>

#include "../ParticlesData.hpp"
#include "Task.hpp"

namespace sphexa
{
    namespace sph
    {

        template <typename T, class Dataset>
        void calcGradhTermsImpl(const Task &t, Dataset &d)
        {
            const size_t n = t.clist.size();
            const int *clist = t.clist.data();


            const T *h = d.h.data();
            const T *ro = d.ro.data();
            const T *sumwh = d.sumwh.data(); // the sum of Xmass weighted by the derivative of the kernel wrt. h

            T *gradh = d.gradh.data();

#pragma omp parallel for
            for (size_t pi = 0; pi < n; pi++)
            {
                const int i = clist[pi];

                gradh[i] = 1.0 + h[i] / ro[i] / 3.0 * sumwh[i];

            }
        }

        template <typename T, class Dataset>
        void calcGradhTerms(const std::vector<Task> &taskList, Dataset &d)
        {
            for (const auto &task : taskList)
            {
                calcGradhTermsImpl<T>(task, d);
            }
        }

    } // namespace sph
} // namespace sphexa
