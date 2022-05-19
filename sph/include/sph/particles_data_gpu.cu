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
 * @brief Contains the object holding all particle data on the GPU
 */

#include "particles_data_gpu.cuh"

namespace sphexa
{

template<typename T, class KeyType>
void DeviceParticlesData<T, KeyType>::resize(size_t size)
{
    double growthRate = 1.01;
    auto   data_      = data();

    auto deallocateVector = [size](auto* devVectorPtr)
    {
        using DevVector = std::decay_t<decltype(*devVectorPtr)>;
        if (devVectorPtr->capacity() < size) { *devVectorPtr = DevVector{}; }
    };

    for (size_t i = 0; i < data_.size(); ++i)
    {
        if (this->isAllocated(i)) { std::visit(deallocateVector, data_[i]); }
    }

    for (size_t i = 0; i < data_.size(); ++i)
    {
        if (this->isAllocated(i))
        {
            std::visit([size, growthRate](auto* arg) { reallocate(*arg, size, growthRate); }, data_[i]);
        }
    }
}

template class DeviceParticlesData<float, unsigned>;
template class DeviceParticlesData<float, uint64_t>;
template class DeviceParticlesData<double, unsigned>;
template class DeviceParticlesData<double, uint64_t>;

} // namespace sphexa
