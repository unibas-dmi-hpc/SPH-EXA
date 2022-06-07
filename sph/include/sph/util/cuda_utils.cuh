#pragma once

#include <cstdio>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <type_traits>

namespace sph
{
namespace cuda
{

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

#define CHECK_CUDA_ERR(errcode) ::sph::cuda::utils::checkErr((errcode), __FILE__, __LINE__, #errcode)

namespace utils
{

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

inline cudaError_t cudaFree() { return cudaSuccess; }

template<typename Ptr, typename... Ptrs>
inline cudaError_t cudaFree(Ptr first, Ptrs... ptrs)
{
    static_assert(std::is_pointer<Ptr>::value, "Parameter passed to cudaFree must be a pointer type");

    const auto ret = ::cudaFree(first);
    if (ret == cudaSuccess) return cudaFree(ptrs...);

    return ret;
}

inline cudaError_t cudaMalloc(size_t) { return cudaSuccess; }

template<typename Ptr, typename... Ptrs>
inline cudaError_t cudaMalloc(size_t bytes, Ptr& devptr, Ptrs&&... ptrs)
{
    static_assert(std::is_pointer<Ptr>::value, "Parameter passed to cudaMalloc must be a pointer type");

    const auto ret = ::cudaMalloc((void**)&devptr, bytes);
    if (ret == cudaSuccess) return cudaMalloc(bytes, ptrs...);

    return ret;
}

} // namespace utils
} // namespace cuda
} // namespace sph
