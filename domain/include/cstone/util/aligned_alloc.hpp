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
 * @brief Aligned allocator
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <cstdlib>
#include <new>
#include <memory>

namespace util
{

// Alignment must be a power of 2 !
template<typename T, unsigned int Alignment>
class AlignedAllocator
{
public:
    typedef T* pointer;
    typedef T const* const_pointer;
    typedef T& reference;
    typedef T const& const_reference;
    typedef T value_type;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;

    template<typename U>
    struct rebind
    {
        typedef AlignedAllocator<U, Alignment> other;
    };

    AlignedAllocator() noexcept {}

    AlignedAllocator(AlignedAllocator const&) noexcept {}

    template<typename U>
    AlignedAllocator(AlignedAllocator<U, Alignment> const&) noexcept
    {
    }

    pointer allocate(size_type n)
    {
        pointer p;
        if (posix_memalign(reinterpret_cast<void**>(&p), Alignment, n * sizeof(T))) throw std::bad_alloc();
        return p;
    }

    void deallocate(pointer p, size_type /*n*/) noexcept { std::free(p); }

    template<typename C, class... Args>
    void construct(C* c, Args&&... args)
    {
        new ((void*)c) C(std::forward<Args>(args)...);
    }

    template<typename C>
    void destroy(C* c)
    {
        c->~C();
    }

    bool operator==(AlignedAllocator const&) const noexcept { return true; }

    bool operator!=(AlignedAllocator const&) const noexcept { return false; }

    template<typename U, unsigned int UAlignment>
    bool operator==(AlignedAllocator<U, UAlignment> const&) const noexcept
    {
        return false;
    }

    template<typename U, unsigned int UAlignment>
    bool operator!=(AlignedAllocator<U, UAlignment> const&) const noexcept
    {
        return true;
    }
};

} // namespace util
