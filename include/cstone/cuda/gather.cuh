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

/*! \file
 * \brief  Exposes gather functionality to reorder arrays with a map
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <cstdint>

#include <thrust/device_vector.h>

template<class T, class I>
class DeviceGather
{
public:
    DeviceGather(const I* map_first, const I* map_last);

    void gather(T* values);

private:
    thrust::device_vector<I> d_ordering_;
    thrust::device_vector<T> d_source_;
    thrust::device_vector<T> d_destination_;
};

extern template class DeviceGather<float,  unsigned>;
extern template class DeviceGather<float,  uint64_t>;
extern template class DeviceGather<double, unsigned>;
extern template class DeviceGather<double, uint64_t>;

