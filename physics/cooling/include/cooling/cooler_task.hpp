//
// Created by Noah Kubli on 30.11.2023.
//

#ifndef SPHEXA_COOLER_TASK_HPP
#define SPHEXA_COOLER_TASK_HPP

#include <algorithm>
#include "cstone/util/tuple_util.hpp"

namespace cooling
{
template<size_t N = 100>
struct task
{
    size_t first;
    size_t n_bins;
    size_t N_last;
    task(size_t first, size_t last)
        : first(first)
    {
        const size_t n_particles = last - first;
        const size_t mod         = n_particles % N;
        const size_t ceil_term   = (mod == 0) ? 0 : 1;
        n_bins                   = n_particles / N + ceil_term;
        N_last                   = n_particles % n_bins;
    }
};

struct block
{
    size_t first;
    size_t len;
    template<size_t N>
    block(size_t i, const task<N>& t)
        : first(t.first + N * i)
        , len((i == t.n_bins - 1) ? t.N_last : N)
    {
    }
};

void copyToBlock(const auto& v, auto& v_block, const block& b)
{
    std::copy_n(v.data() + b.first, b.len, v_block.data());
}

void copyFromBlock(const auto& v_block, auto& v, const block& b)
{
    std::copy_n(v_block.data(), b.len, v.data() + b.first);
}

template<typename ...T>
auto getBlockPointers(const std::tuple<T*...>& particle, const block& b)
{
    auto f = [&](auto*... args) { return std::make_tuple(args + b.first...); };
    return std::apply(f, particle);
};

template <typename T1, typename ...T>
void multiply_in_place(const T1 *factor, std::tuple<T*...> t, const size_t len)
{
    auto f = [&](auto *arg)
    {
        for (size_t i = 0; i < len; i++)
            arg[i] *= factor[i];
    };
    util::for_each_tuple(f, t);
}

template <typename T1, typename ...T>
void divide_in_place(const T1 *factor, std::tuple<T*...> t, const size_t len)
{
    auto f = [&](auto *arg)
    {
        for (size_t i = 0; i < len; i++)
            arg[i] /= factor[i];
    };
    util::for_each_tuple(f, t);
}

}
#endif // SPHEXA_COOLER_TASK_HPP
