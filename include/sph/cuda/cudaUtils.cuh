#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <type_traits>

namespace sphexa
{
namespace sph
{
namespace cuda
{
#define CHECK_CUDA_ERR(errcode) sphexa::sph::cuda::utils::checkErr((errcode), __FILE__, __LINE__, #errcode);

namespace utils
{

inline void checkErr(cudaError_t err, const char *filename, int lineno, const char *funcName)
{
    if (err != cudaSuccess)
    {
        const char *errName = cudaGetErrorName(err);
        const char *errStr = cudaGetErrorString(err);
        fprintf(stderr, "CUDA Error at %s:%d. Function %s returned err %d: %s - %s\n", filename, lineno, funcName, err, errName, errStr);
    }
}

inline cudaError_t cudaFree() { return cudaSuccess; }

template <typename Ptr, typename... Ptrs>
inline cudaError_t cudaFree(Ptr first, Ptrs... ptrs)
{
    static_assert(std::is_pointer<Ptr>::value, "Parameter passed to cudaFree must be a pointer type");

    const auto ret = ::cudaFree(first);
    if (ret == cudaSuccess) return cudaFree(ptrs...);

    return ret;
}

inline cudaError_t cudaMalloc(size_t) { return cudaSuccess; }

template <typename Ptr, typename... Ptrs>
inline cudaError_t cudaMalloc(size_t bytes, Ptr &devptr, Ptrs &&... ptrs)
{
    static_assert(std::is_pointer<Ptr>::value, "Parameter passed to cudaMalloc must be a pointer type");

    const auto ret = ::cudaMalloc((void **)&devptr, bytes);
    if (ret == cudaSuccess) return cudaMalloc(bytes, ptrs...);

    return ret;
}

} // namespace utils



template<typename T>
struct DeviceLinearOctree
{
    int size;
    int *ncells;
    int *cells;
    int *localPadding;
    int *localParticleCount;
    T *xmin, *xmax, *ymin, *ymax, *zmin, *zmax;
};

template<typename T>
void mapLinearOctreeToDevice(const LinearOctree<T> &o, DeviceLinearOctree<T> &d_o)
{
    size_t size_int = o.size * sizeof(int);
    size_t size_T = o.size * sizeof(T);

    d_o.size = o.size;

    CHECK_CUDA_ERR(utils::cudaMalloc(size_int * 8, d_o.cells));
    CHECK_CUDA_ERR(utils::cudaMalloc(size_int, d_o.ncells, d_o.localPadding, d_o.localParticleCount));
    CHECK_CUDA_ERR(utils::cudaMalloc(size_T, d_o.xmin, d_o.xmax, d_o.ymin, d_o.ymax, d_o.zmin, d_o.zmax));

    CHECK_CUDA_ERR(cudaMemcpy(d_o.cells, o.cells.data(), size_int * 8, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_o.ncells, o.ncells.data(), size_int, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_o.localPadding, o.localPadding.data(), size_int, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_o.localParticleCount, o.localParticleCount.data(), size_int, cudaMemcpyHostToDevice));

    CHECK_CUDA_ERR(cudaMemcpy(d_o.xmin, o.xmin.data(), size_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_o.xmax, o.xmax.data(), size_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_o.ymin, o.ymin.data(), size_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_o.ymax, o.ymax.data(), size_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_o.zmin, o.zmin.data(), size_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_o.zmax, o.zmax.data(), size_T, cudaMemcpyHostToDevice));
}

template<typename T>
void unmapLinearOctreeFromDevice(DeviceLinearOctree<T> &d_o)
{
    CHECK_CUDA_ERR(utils::cudaFree(d_o.cells));
    CHECK_CUDA_ERR(utils::cudaFree(d_o.ncells, d_o.localPadding, d_o.localParticleCount));
    CHECK_CUDA_ERR(utils::cudaFree(d_o.xmin, d_o.xmax, d_o.ymin, d_o.ymax, d_o.zmin, d_o.zmax));
}


} // namespace cuda
} // namespace sph
} // namespace sphexa
