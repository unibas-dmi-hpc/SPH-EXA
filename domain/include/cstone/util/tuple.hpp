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
 * @brief  Wrapper around different types of tuples compatible with device code
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <tuple>

#if defined(__CUDACC__) || defined(__HIPCC__)
#include <thrust/tuple.h>
#endif

#include "cstone/cuda/annotation.hpp"

#if defined(__CUDACC__) || defined(__HIPCC__)

namespace util
{

template<class... Ts>
using tuple = thrust::tuple<Ts...>;

template<size_t N, class T>
constexpr __host__ __device__ auto get(T&& tup) noexcept
{
    return thrust::get<N>(tup);
}

template<class... Ts>
constexpr __host__ __device__ tuple<Ts&...> tie(Ts&... args) noexcept
{
    return thrust::tuple<Ts&...>(args...);
}

} // namespace util

//! @brief specializations of tuple traits in std:: namespace to make structured binding work with thrust tuples
namespace std
{

template<size_t N, class... Ts>
struct tuple_element<N, thrust::tuple<Ts...>>
{
    typedef typename thrust::tuple_element<N, thrust::tuple<Ts...>>::type type;
};

template<class... Ts>
struct tuple_size<thrust::tuple<Ts...>>
{
    static const int value = thrust::tuple_size<thrust::tuple<Ts...>>::value;
};

} // namespace std

#else

namespace util
{

template<class... Ts>
using tuple = std::tuple<Ts...>;

template<std::size_t N, class T>
constexpr auto get(T&& tup) noexcept
{
    return std::get<N>(tup);
}

template<class... Ts>
constexpr tuple<Ts&...> tie(Ts&... args) noexcept
{
    return std::tuple<Ts&...>(args...);
}

} // namespace util
#endif

namespace util
{

template<class Tuple>
struct TuplePlusImpl
{
    template<std::size_t... Is>
    HOST_DEVICE_FUN Tuple operator()(const Tuple& a, const Tuple& b, std::index_sequence<Is...>)
    {
        return Tuple((util::get<Is>(a) + util::get<Is>(b))...);
    }
};

/*! @brief generic tuple addition functor that works for both thrust and std tuples
 *
 * @tparam Tuple   the kind of tuple to be added, e.g. thrust::tuple<int, double>
 */
template<class Tuple>
struct TuplePlus
{
    HOST_DEVICE_FUN Tuple operator()(const Tuple& a, const Tuple& b)
    {
        return TuplePlusImpl<Tuple>{}(a, b, std::make_index_sequence<std::tuple_size_v<Tuple>>{});
    }
};

} // namespace util