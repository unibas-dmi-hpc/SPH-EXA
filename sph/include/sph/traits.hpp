/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
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
 * @brief Traits classes for ParticlesData to abstract and manage GPU device acceleration behavior
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <type_traits>

#include "cstone/domain/domain_traits.hpp"

template<class T>
class pinned_allocator;

namespace sphexa
{

using cstone::CpuTag;
using cstone::GpuTag;

template<class AccType>
struct HaveGpu : public stl::integral_constant<int, std::is_same_v<AccType, GpuTag>>
{
};

//! @brief stub for use in CPU code
template<class T, class KeyType>
struct DeviceDataFacade
{
    void resize(size_t) {}

    template<class... Ts>
    void setConserved(Ts...)
    {
    }

    template<class... Ts>
    void setDependent(Ts...)
    {
    }

    template<class... Ts>
    void release(Ts...)
    {
    }

    template<class... Ts>
    void acquire(Ts...)
    {
    }

    inline static constexpr std::array fieldNames{0};
};

template<class T, class KeyType>
class DeviceParticlesData;

namespace detail
{

//! @brief The type member of this trait evaluates to CpuCaseType if Accelerator == CpuTag and GpuCaseType otherwise
template<class Accelerator, template<class...> class CpuCaseType, template<class...> class GpuCaseType, class = void>
struct AccelSwitchType
{
};

template<class Accelerator, template<class...> class CpuCaseType, template<class...> class GpuCaseType>
struct AccelSwitchType<Accelerator, CpuCaseType, GpuCaseType, std::enable_if_t<!HaveGpu<Accelerator>{}>>
{
    template<class... Args>
    using type = CpuCaseType<Args...>;
};

template<class Accelerator, template<class...> class CpuCaseType, template<class...> class GpuCaseType>
struct AccelSwitchType<Accelerator, CpuCaseType, GpuCaseType, std::enable_if_t<HaveGpu<Accelerator>{}>>
{
    template<class... Args>
    using type = GpuCaseType<Args...>;
};

} // namespace detail

//! @brief Just a facade on the CPU, DeviceParticlesData on the GPU
template<class Accelerator, class T, class KeyType>
using DeviceData_t =
    typename detail::AccelSwitchType<Accelerator, DeviceDataFacade, DeviceParticlesData>::template type<T, KeyType>;

//! @brief std::allocator on the CPU, pinned_allocator on the GPU
template<class Accelerator, class T>
using PinnedAlloc_t = typename detail::AccelSwitchType<Accelerator, std::allocator, pinned_allocator>::template type<T>;

} // namespace sphexa
