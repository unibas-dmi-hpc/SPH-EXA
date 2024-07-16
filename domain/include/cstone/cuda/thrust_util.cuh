/*! @file
 * @brief Utilities for Thrust device vectors for use in .cu translation units only
 */

#pragma once

#include <thrust/device_vector.h>

template<class T, class Alloc>
T* rawPtr(thrust::device_vector<T, Alloc>& p)
{
    return thrust::raw_pointer_cast(p.data());
}

template<class T, class Alloc>
const T* rawPtr(const thrust::device_vector<T, Alloc>& p)
{
    return thrust::raw_pointer_cast(p.data());
}
