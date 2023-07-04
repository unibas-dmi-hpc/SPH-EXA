/*
 * MIT License
 *
 * Copyright (c) 2022 CSCS, ETH Zurich
 *               2021 University of Basel
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
 * @brief reductions for different observables on the GPU
 *
 * @author Lukas Schmidt
 */

#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform_reduce.h>

#include "cstone/util/tuple.hpp"
#include "gpu_reductions.h"

namespace sphexa
{

using cstone::Box;
using cstone::Vec3;
using thrust::get;

//!@brief functor for Kelvin-Helmholtz growth rate
template<class Tc, class Tv, class Tm>
struct GrowthRate
{
    HOST_DEVICE_FUN
    thrust::tuple<double, double, double> operator()(const thrust::tuple<Tc, Tc, Tv, Tm, Tm>& p)
    {
        Tc xi   = get<0>(p);
        Tc yi   = get<1>(p);
        Tv vyi  = get<2>(p);
        Tm xmi  = get<3>(p);
        Tm kxi  = get<4>(p);
        Tc voli = xmi / kxi;
        Tc aux;

        if (yi < ybox * Tc(0.5)) { aux = std::exp(-4.0 * M_PI * std::abs(yi - 0.25)); }
        else { aux = std::exp(-4.0 * M_PI * std::abs(ybox - yi - 0.25)); }
        Tc si = vyi * voli * std::sin(4.0 * M_PI * xi) * aux;
        Tc ci = vyi * voli * std::cos(4.0 * M_PI * xi) * aux;
        Tc di = voli * aux;

        return thrust::make_tuple(si, ci, di);
    }

    Tc ybox;
};

template<class Tc, class Tv, class Tm>
std::tuple<double, double, double> gpuGrowthRate(const Tc* x, const Tc* y, const Tv* vy, const Tm* xm, const Tm* kx,
                                                 const cstone::Box<Tc>& box, size_t startIndex, size_t endIndex)
{
    auto it1 = thrust::make_zip_iterator(
        thrust::make_tuple(x + startIndex, y + startIndex, vy + startIndex, xm + startIndex, kx + startIndex));

    auto it2 = thrust::make_zip_iterator(
        thrust::make_tuple(x + endIndex, y + endIndex, vy + endIndex, xm + endIndex, kx + endIndex));

    auto plus = util::TuplePlus<thrust::tuple<double, double, double>>{};
    auto init = thrust::make_tuple(0.0, 0.0, 0.0);

    auto result = thrust::transform_reduce(thrust::device, it1, it2, GrowthRate<Tc, Tm, Tv>{box.ly()}, init, plus);

    return {get<0>(result), get<1>(result), get<2>(result)};
}

#define GROWTH_GPU(Tc, Tv, Tm)                                                                                         \
    template std::tuple<double, double, double> gpuGrowthRate(const Tc* x, const Tc* y, const Tv* vy, const Tm* xm,    \
                                                              const Tm* kx, const cstone::Box<Tc>& box,                \
                                                              size_t startIndex, size_t endIndex)

GROWTH_GPU(double, double, double);
GROWTH_GPU(double, double, float);
GROWTH_GPU(double, float, float);
GROWTH_GPU(float, float, float);

//!@brief functor for the machSquare summing
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

//!@brief functor to count particles that belong to the cloud
template<class Tc, class Tt, class Tm>
struct Survivors
{
    HOST_DEVICE_FUN
    double operator()(const thrust::tuple<Tt, Tc, Tc, Tm>& p)
    {
        size_t isCloud;
        Tt     temp = get<0>(p);
        Tc     kx   = get<1>(p);
        Tc     xm   = get<2>(p);
        Tm     m    = get<3>(p);
        Tc     rhoi = kx / xm * m;
        if (rhoi >= 0.64 * rhoBubble && temp <= 0.9 * tempWind) { isCloud = 1; }
        else { isCloud = 0; }
        return isCloud;
    }

    double rhoBubble;
    double tempWind;
};
template<class Tc, class Tt, class Tm>
size_t survivorsGpu(const Tt* temp, const Tt* kx, const Tc* xmass, const Tm* m, double rhoBubble, double tempWind,
                    size_t first, size_t last)
{
    auto it1 = thrust::make_zip_iterator(thrust::make_tuple(temp + first, kx + first, xmass + first, m + first));
    auto it2 = thrust::make_zip_iterator(thrust::make_tuple(temp + last, kx + last, xmass + last, m + last));

    auto plus = thrust::plus<size_t>{};

    size_t localSurvivors = 0;

    localSurvivors = thrust::transform_reduce(thrust::device, it1, it2, Survivors<Tc, Tt, Tm>{rhoBubble, tempWind},
                                              localSurvivors, plus);

    return localSurvivors;
}

#define SURVIVORS(Tc, Tt, Tm)                                                                                          \
    template size_t survivorsGpu(const Tt* temp, const Tt* kx, const Tc* xmass, const Tm* m, double, double, size_t,   \
                                 size_t)

SURVIVORS(double, double, double);
SURVIVORS(double, double, float);
SURVIVORS(double, float, float);
SURVIVORS(float, float, float);

} // namespace sphexa
