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
