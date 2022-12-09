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

#include "cstone/util/tuple.hpp"

#include "conserved_gpu.h"

namespace sphexa
{

using cstone::Vec3;
using thrust::get;

/*! @brief Functor to compute kinetic and internal energy and linear and angular momentum
 *
 * @tparam Tc   type of x,y,z coordinates
 * @tparam Tm   type of mass
 * @tparam Tv   type of velocities
 * @tparam Tt   type of temperature
 */
template<class Tc, class Tm, class Tv, class Tt>
struct EMom
{
    /*! @brief compute energies and momenta for a single particle
     *
     * @param p   Tuple<x,y,z,m,vx,vy,vz,temp> with data for one particle
     * @return    Tuple<kinetic energy, internal energy, linear momentum, angular momentum>
     */
    HOST_DEVICE_FUN
    thrust::tuple<double, double, Vec3<double>, Vec3<double>>
    operator()(const thrust::tuple<Tc, Tc, Tc, Tm, Tv, Tv, Tv, Tt>& p)
    {
        Vec3<double> X{get<0>(p), get<1>(p), get<2>(p)};
        Vec3<double> V{get<4>(p), get<5>(p), get<6>(p)};
        Tm           m    = get<3>(p);
        Tt           temp = get<7>(p);
        return thrust::make_tuple(m * norm2(V), cv * m * temp, double(m) * V, double(m) * cross(X, V));
    }

    Tt cv;
};

template<class Tc, class Tv, class Tt, class Tm>
std::tuple<double, double, Vec3<double>, Vec3<double>>
conservedQuantitiesGpu(Tt cv, const Tc* x, const Tc* y, const Tc* z, const Tv* vx, const Tv* vy, const Tv* vz,
                       const Tt* temp, const Tm* m, size_t first, size_t last)
{
    auto it1 = thrust::make_zip_iterator(thrust::make_tuple(x + first, y + first, z + first, m + first, vx + first,
                                                            vy + first, vz + first, temp + first));
    auto it2 = thrust::make_zip_iterator(
        thrust::make_tuple(x + last, y + last, z + last, m + last, vx + last, vy + last, vz + last, temp + last));

    auto plus = util::TuplePlus<thrust::tuple<double, double, Vec3<double>, Vec3<double>>>{};
    auto init = thrust::make_tuple(0.0, 0.0, Vec3<double>{0, 0, 0}, Vec3<double>{0, 0, 0});

    //! apply EMom to each particle and reduce results into a single sum
    auto result = thrust::transform_reduce(thrust::device, it1, it2, EMom<Tc, Tm, Tv, Tt>{cv}, init, plus);

    return {0.5 * get<0>(result), get<1>(result), get<2>(result), get<3>(result)};
}

#define CONSERVED_Q_GPU(Tc, Tv, Tt, Tm)                                                                                \
    template std::tuple<double, double, Vec3<double>, Vec3<double>> conservedQuantitiesGpu(                            \
        Tt, const Tc* x, const Tc* y, const Tc* z, const Tv* vx, const Tv* vy, const Tv* vz, const Tt* temp,           \
        const Tm* m, size_t, size_t)

CONSERVED_Q_GPU(double, double, double, double);
CONSERVED_Q_GPU(double, double, double, float);
CONSERVED_Q_GPU(double, float, double, float);
CONSERVED_Q_GPU(float, float, float, float);

template<class Tc, class Tv>
struct MachSquareSum
{
    HOST_DEVICE_FUN
    double operator()(const thrust::tuple<Tv, Tv, Tv, Tc>& p)
    {
        Vec3<double> V{get<0>(p), get<1>(p), get<2>(p)};
        Tc           c = get<3>(p);
        return norm2(V) / (double(c) * c);
    }
};

template<class Tc, class Tv>
double machSquareSumGpu(const Tv* vx, const Tv* vy, const Tv* vz, const Tc* c, size_t first, size_t last)
{
    auto it1 = thrust::make_zip_iterator(thrust::make_tuple(vx + first, vy + first, vz + first, c + first));
    auto it2 = thrust::make_zip_iterator(thrust::make_tuple(vx + last, vy + last, vz + last, c + last));

    auto   plus               = thrust::plus<double>{};
    double localMachSquareSum = 0.0;

    localMachSquareSum =
        thrust::transform_reduce(thrust::device, it1, it2, MachSquareSum<Tc, Tv>{}, localMachSquareSum, plus);

    return localMachSquareSum;
}

#define MACH_GPU(Tc, Tv)                                                                                               \
    template double machSquareSumGpu(const Tv* vx, const Tv* vy, const Tv* vz, const Tc* c, size_t, size_t)

MACH_GPU(double, double);
MACH_GPU(double, float);
MACH_GPU(float, float);

} // namespace sphexa
