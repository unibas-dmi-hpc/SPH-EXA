/*! @file
 * @brief  Encapsulation around a std::vector like GPU-resident device vector, suitable for use in .cpp files
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <memory>

#include "cstone/util/array.hpp"

namespace cstone
{

//! @brief Encapsulation for a device vector with a minimal interface that hides any GPU/CUDA headers from CPU code
template <class T>
class DeviceVector
{
public:
    DeviceVector();

    ~DeviceVector();

    T* data();
    const T* data() const;

    void resize(std::size_t size);

    void reserve(std::size_t size);

    std::size_t size() const;
    std::size_t capacity() const;

private:
   class Impl;
   std::unique_ptr<Impl> impl_;
};

extern template class DeviceVector<char>;
extern template class DeviceVector<uint8_t>;
extern template class DeviceVector<int>;
extern template class DeviceVector<unsigned>;
extern template class DeviceVector<uint64_t>;
extern template class DeviceVector<float>;
extern template class DeviceVector<double>;
extern template class DeviceVector<util::array<int, 2>>;
extern template class DeviceVector<util::array<float, 3>>;
extern template class DeviceVector<util::array<double, 3>>;
extern template class DeviceVector<util::array<float, 4>>;
extern template class DeviceVector<util::array<double, 4>>;

} // namespace cstone
