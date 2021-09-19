#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <type_traits>

namespace sphexa
{
namespace sph
{
namespace cuda
{
#define CHECK_CUDA_ERR(errcode) sphexa::sph::cuda::utils::checkErr((errcode), __FILE__, __LINE__, #errcode)

namespace utils
{

inline void checkErr(hipError_t err, const char *filename, int lineno, const char *funcName)
{
    if (err != hipSuccess)
    {
        const char *errName = hipGetErrorName(err);
        const char *errStr = hipGetErrorString(err);
        fprintf(stderr, "CUDA Error at %s:%d. Function %s returned err %d: %s - %s\n", filename, lineno, funcName, err, errName, errStr);
    }
}

inline hipError_t hipFree() { return hipSuccess; }

template <typename Ptr, typename... Ptrs>
inline hipError_t hipFree(Ptr first, Ptrs... ptrs)
{
    static_assert(std::is_pointer<Ptr>::value, "Parameter passed to hipFree must be a pointer type");

    const auto ret = ::hipFree(first);
    if (ret == hipSuccess) return hipFree(ptrs...);

    return ret;
}

inline hipError_t hipMalloc(size_t) { return hipSuccess; }

template <typename Ptr, typename... Ptrs>
inline hipError_t hipMalloc(size_t bytes, Ptr &devptr, Ptrs &&... ptrs)
{
    static_assert(std::is_pointer<Ptr>::value, "Parameter passed to hipMalloc must be a pointer type");

    const auto ret = ::hipMalloc((void **)&devptr, bytes);
    if (ret == hipSuccess) return hipMalloc(bytes, ptrs...);

    return ret;
}

} // namespace utils
} // namespace cuda
} // namespace sph
} // namespace sphexa
