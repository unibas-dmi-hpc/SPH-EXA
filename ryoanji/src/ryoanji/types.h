#pragma once

#define CUDA_SAFE_CALL(err) cudaSafeCall(err, __FILE__, __LINE__)

#include <cassert>
#include <thrust/device_vector.h>

#include "cstone/util/array.hpp"

const int P = 4;

const int NTERM = P * (P + 1) * (P + 2) / 6; // 20 for P=4
const int NVEC4 = (NTERM - 1) / 4 + 1;       // 5 for P=4
typedef util::array<float, 3> fvec3;
typedef util::array<float, 4> fvec4;
typedef util::array<float, NTERM> fvecP;

namespace ryoanji
{

struct GpuConfig
{
    //! @brief number of threads per warp, adapt to actual hardware
    static constexpr int warpSize = 32;

    static_assert(warpSize == 32 || warpSize == 64, "warp size has to be 32 or 64");

    //! @brief log2(warpSize)
    static constexpr int warpSizeLog2 = (warpSize == 32) ? 5 : 6;

    //! @brief number of multiprocessors, set based on cudaGetDeviceProp
    inline static int smCount = 56;

    /*! @brief integer type for representing a thread mask, e.g. return value of __ballot_sync()
     *
     * This will automatically pick the right type based on the warpSize choice. Do not adapt.
     */
    using ThreadMask = std::conditional_t<warpSize == 32, unsigned, uint64_t>;
};

//! Center and radius of bounding box
struct Box
{
    fvec3 X; //!< Box center
    float R; //!< Box radius
};

//! Min & max bounds of bounding box
struct Bounds
{
    fvec3 Xmin; //!< Minimum value of coordinates
    fvec3 Xmax; //!< Maximum value of coordinates
};

template<class T>
T* rawPtr(thrust::device_ptr<T> p)
{
    return thrust::raw_pointer_cast(p);
}

class CellData
{
private:
    static const int CHILD_SHIFT = 29;
    static const int CHILD_MASK  = ~(0x7U << CHILD_SHIFT);
    static const int LEVEL_SHIFT = 27;
    static const int LEVEL_MASK  = ~(0x1FU << LEVEL_SHIFT);
    uint4 data;

public:
    CellData() = default;

    __host__ __device__ CellData(const unsigned int level, const unsigned int parent, const unsigned int body,
                                 const unsigned int nbody, const unsigned int child = 0, const unsigned int nchild = 1)
    {
        unsigned parentPack = parent | (level << LEVEL_SHIFT);
        unsigned childPack  = child | ((nchild - 1) << CHILD_SHIFT);
        data                = make_uint4(parentPack, childPack, body, nbody);
    }

    __host__ __device__ CellData(uint4 data_)
        : data(data_)
    {
    }

    __host__ __device__ int level() const { return data.x >> LEVEL_SHIFT; }
    __host__ __device__ int parent() const { return data.x & LEVEL_MASK; }
    __host__ __device__ int child() const { return data.y & CHILD_MASK; }
    __host__ __device__ int nchild() const { return (data.y >> CHILD_SHIFT) + 1; }
    __host__ __device__ int body() const { return data.z; }
    __host__ __device__ int nbody() const { return data.w; }
    __host__ __device__ bool isLeaf() const { return data.y == 0; }
    __host__ __device__ bool isNode() const { return !isLeaf(); }

    __host__ __device__ void setParent(unsigned parent) { data.x = parent | (level() << LEVEL_SHIFT); }
    __host__ __device__ void setChild(unsigned child) { data.y = child | (nchild() - 1 << CHILD_SHIFT); }
    __host__ __device__ void setBody(unsigned body_) { data.z = body_; }
    __host__ __device__ void setNBody(unsigned nbody_) { data.w = nbody_; }
};

inline __host__ __device__ fvec3 make_fvec3(fvec4 v)
{
    fvec3 data;
    data[0] = v[0];
    data[1] = v[1];
    data[2] = v[2];
    return data;
}

static void kernelSuccess(const char kernel[] = "kernel")
{
    cudaDeviceSynchronize();
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "%s launch failed: %s\n", kernel, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

static __forceinline__ void cudaSafeCall(cudaError err, const char* file, const int line)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

} // namespace ryoanji
