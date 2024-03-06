//
// Created by Noah Kubli on 01.03.2024.
//

#pragma once

#include <functional>
#include <tuple>

namespace cooling
{
//! @brief For Tuples A, B ... call f(a1, b1 ...), f(a2, b2 ...)
template<typename... Tuples, typename F>
requires(std::tuple_size_v<std::decay_t<Tuples>> == ...) void for_each_tuples(F&& f, Tuples&&... tuples)
{
    auto f_i = [&](auto I) { return f(std::get<I>(std::forward<Tuples>(tuples))...); };

    auto iterate_each = [&f_i]<size_t... Is>(std::index_sequence<Is...>)
    {
        (f_i(std::integral_constant<size_t, Is>{}), ...);
    };

    constexpr size_t n_elements = std::min({std::tuple_size_v<std::decay_t<Tuples>>...});
    iterate_each(std::make_index_sequence<n_elements>{});
}

} // namespace cooling
