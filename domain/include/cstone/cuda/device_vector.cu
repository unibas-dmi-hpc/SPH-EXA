/*! @file
 * @brief  Encapsulation around a std::vector like GPU-resident device vector, suitable for use in .cpp files
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <thrust/device_vector.h>
#include <cuda_runtime.h>

#include "cstone/util/noinit_thrust.cuh"

#include "device_vector.h"

namespace cstone
{

template<class T>
class DeviceVector<T>::Impl
{
public:
    Impl() {}

    T* data() { return thrust::raw_pointer_cast(data_.data()); }
    const T* data() const { return thrust::raw_pointer_cast(data_.data()); }

    void resize(std::size_t size) { data_.resize(size); }

    void reserve(std::size_t size) { data_.reserve(size); };

    std::size_t size() const { return data_.size(); };
    std::size_t capacity() const { return data_.capacity(); };

private:
    thrust::device_vector<T, util::uninitialized_allocator<T>> data_;
};

template<class T>
DeviceVector<T>::DeviceVector()
    : impl_(new Impl())
{
}

template<class T>
DeviceVector<T>::~DeviceVector() = default;

template<class T>
T* DeviceVector<T>::data() { return impl_->data(); };

template<class T>
const T* DeviceVector<T>::data() const { return impl_->data(); };

template<class T>
void DeviceVector<T>::resize(std::size_t size) { impl_->resize(size); };

template<class T>
void DeviceVector<T>::reserve(std::size_t size) { impl_->reserve(size); };

template<class T>
std::size_t DeviceVector<T>::size() const { return impl_->size(); };

template<class T>
std::size_t DeviceVector<T>::capacity() const { return impl_->capacity(); };

template class DeviceVector<char>;
template class DeviceVector<int>;
template class DeviceVector<unsigned>;
template class DeviceVector<uint64_t>;
template class DeviceVector<float>;
template class DeviceVector<double>;
template class DeviceVector<util::array<int, 2>>;
template class DeviceVector<util::array<float, 3>>;
template class DeviceVector<util::array<double, 3>>;
template class DeviceVector<util::array<float, 4>>;
template class DeviceVector<util::array<double, 4>>;

} // namespace cstone
