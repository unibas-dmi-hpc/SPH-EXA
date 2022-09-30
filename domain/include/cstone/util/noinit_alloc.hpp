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
 * @brief Allocator adaptor to prevent value initialization
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <cstdlib>
#include <new>
#include <memory>

namespace util
{

/*! @brief Allocator adaptor to prevent value initialization from happening.
 *
 * @tparam  Alloc   the underlying allocator type to adapt to default initialization
 *
 * This works by intercepting construct() calls to convert value initialization into
 * default initialization. The main motivation for this is to avoid touching the memory
 * when constructing a std::vector such that we can cleanly apply first-touch policy
 * to NUMA placement. By default, only one thread initializes the values of a std::vector,
 * making it impossible to achieve optimal NUMA placement.
 */
template<typename T, typename Alloc = std::allocator<T>>
class DefaultInitAdaptor : public Alloc
{
    using AllocType = std::allocator_traits<Alloc>;

public:
    template<typename U>
    struct rebind
    {
        using other = DefaultInitAdaptor<U, typename AllocType::template rebind_alloc<U>>;
    };

    using Alloc::Alloc;

    //! @brief construct with no arguments given, e.g. std::vector<T, [ThisClass]> vec(10);
    template<typename U>
    void construct(U* ptr) noexcept(std::is_nothrow_default_constructible<U>::value)
    {
        // default initialization here, instead of U(), prevents zeroing
        ::new (static_cast<void*>(ptr)) U;
    }

    /*! @brief construct with arguments given, e.g. std::vector<T, [ThisClass]> vec(10, 0);
     *
     * If the init value is explicitly specified, it will be forwarded to the underlying allocator.
     * In this case, element initialization takes places as normal.
     */
    template<typename U, typename... Args>
    void construct(U* ptr, Args&&... args)
    {
        AllocType::construct(static_cast<Alloc&>(*this), ptr, std::forward<Args>(args)...);
    }
};

} // namespace util
