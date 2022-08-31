/*
 * MIT License
 *
 * Copyright (c) 2022 CSCS, ETH Zurich
 *               2022 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief  Energy and momentum reductions on the GPU
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform_reduce.h>

#include "conserved_gpu.h"

namespace sphexa
{

using cstone::Vec3;

template<class Tm, class Tv, class Tu>
struct EKinInt
{
    HOST_DEVICE_FUN
    thrust::tuple<double, double> operator()(const thrust::tuple<Tm, Tv, Tv, Tv, Tu>& p)
    {
        cstone::Vec3<Tv> V{thrust::get<1>(p), thrust::get<2>(p), thrust::get<3>(p)};
        Tm m = thrust::get<0>(p);
        Tu u = thrust::get<4>(p);
        return thrust::make_tuple(m * norm2(V), m * u);
    }
};

template<class T1, class T2>
struct TuplePlus
{
    HOST_DEVICE_FUN thrust::tuple<T1, T2> operator()(const thrust::tuple<T1, T2>& a, const thrust::tuple<T1, T2>& b)
    {
        return thrust::make_tuple(thrust::get<0>(a) + thrust::get<0>(b), thrust::get<1>(a) + thrust::get<1>(b));
    }
};

template<class Tc, class Tv, class Tu, class Tm>
std::tuple<Tc, Tc, Vec3<Tc>, Vec3<Tc>> conservedQuantitiesGpu(const Tc* x,
                                                              const Tc* y,
                                                              const Tc* z,
                                                              const Tv* vx,
                                                              const Tv* vy,
                                                              const Tv* vz,
                                                              const Tu* u,
                                                              const Tm* m,
                                                              size_t first,
                                                              size_t last)
{
    auto it1 = thrust::make_zip_iterator(thrust::make_tuple(m + first, vx + first, vy + first, vz + first, u + first));
    auto it2 = thrust::make_zip_iterator(thrust::make_tuple(m + last, vx + last, vy + last, vz + last, u + last));

    auto plus = TuplePlus<double, double>{};
    auto init = thrust::make_tuple(0.0, 0.0);
    auto result = thrust::transform_reduce(thrust::device, it1, it2, EKinInt<Tm, Tv, Tu>{}, init, plus);
    return {0.5 * thrust::get<0>(result), thrust::get<1>(result), Vec3<Tc>{0, 0, 0}, Vec3<Tc>{0, 0, 0}};
}

#define CONSERVED_Q_GPU(Tc, Tv, Tu, Tm)                                                                                \
    template std::tuple<Tc, Tc, Vec3<Tc>, Vec3<Tc>> conservedQuantitiesGpu(const Tc* x, const Tc* y, const Tc* z,      \
                                                                           const Tv* vx, const Tv* vy, const Tv* vz,   \
                                                                           const Tu* u, const Tm* m, size_t, size_t)

CONSERVED_Q_GPU(double, double, double, double);
CONSERVED_Q_GPU(double, double, double, float);
CONSERVED_Q_GPU(double, float, double, float);
CONSERVED_Q_GPU(float, float, float, float);

} // namespace sphexa
