#pragma once

#include <cstdio>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <type_traits>

#include <thrust/device_vector.h>

template<class ThrustVec>
typename ThrustVec::value_type* rawPtr(ThrustVec& p)
{
    assert(p.size() && "cannot get pointer to unallocated device vector memory");
    return thrust::raw_pointer_cast(p.data());
}

template<class ThrustVec>
const typename ThrustVec::value_type* rawPtr(const ThrustVec& p)
{
    assert(p.size() && "cannot get pointer to unallocated device vector memory");
    return thrust::raw_pointer_cast(p.data());
}

namespace sph
{
namespace cuda
{

#ifdef __CUDA_ARCH__
//! @brief compute atomic min for floats using integer operations
__device__ __forceinline__ float atomicMinFloat(float* addr, float value)
{
    float old;
    old = (value >= 0) ? __int_as_float(atomicMin((int*)addr, __float_as_int(value)))
                       : __uint_as_float(atomicMax((unsigned int*)addr, __float_as_uint(value)));

    return old;
}
#endif

} // namespace cuda

inline void checkErr(cudaError_t err, const char* filename, int lineno, const char* funcName)
{
    if (err != cudaSuccess)
    {
        const char* errName = cudaGetErrorName(err);
        const char* errStr  = cudaGetErrorString(err);
        fprintf(stderr,
                "CUDA Error at %s:%d. Function %s returned err %d: %s - %s\n",
                filename,
                lineno,
                funcName,
                err,
                errName,
                errStr);
    }
}

#define CHECK_CUDA_ERR(errcode) ::sph::checkErr((errcode), __FILE__, __LINE__, #errcode)

} // namespace sph
