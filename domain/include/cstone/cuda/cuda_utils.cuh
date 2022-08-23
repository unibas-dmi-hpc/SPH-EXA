#pragma once

#include <type_traits>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>

#include "errorcheck.cuh"

template<class T, class Alloc>
T* rawPtr(thrust::device_vector<T, Alloc>& p)
{
    assert(p.size() && "cannot get pointer to unallocated device vector memory");
    return thrust::raw_pointer_cast(p.data());
}

template<class T, class Alloc>
const T* rawPtr(const thrust::device_vector<T, Alloc>& p)
{
    assert(p.size() && "cannot get pointer to unallocated device vector memory");
    return thrust::raw_pointer_cast(p.data());
}

template<class T>
void memcpyH2D(const T* src, size_t n, T* dest)
{
    checkGpuErrors(cudaMemcpy(dest, src, sizeof(T) * n, cudaMemcpyHostToDevice));
}

template<class T>
void memcpyD2H(const T* src, size_t n, T* dest)
{
    checkGpuErrors(cudaMemcpy(dest, src, sizeof(T) * n, cudaMemcpyDeviceToHost));
}
