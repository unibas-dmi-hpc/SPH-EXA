/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief  Ryoanji tree cell types and utilities
 *
 * @author Rio Yokota <rioyokota@gsic.titech.ac.jp>
 */

#pragma once

#include <cassert>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "cstone/cuda/annotation.hpp"
#include "cstone/cuda/errorcheck.cuh"
#include "cstone/primitives/stl.hpp"
#include "cstone/util/array.hpp"

const int P = 4;

const int NTERM = P * (P + 1) * (P + 2) / 6; // 20 for P=4
const int NVEC4 = (NTERM - 1) / 4 + 1;       // 5 for P=4
typedef util::array<float, 3> fvec3;
typedef util::array<float, 4> fvec4;
typedef util::array<float, NTERM> fvecP;

template<class T>
using Vec3 = util::array<T, 3>;

template<class T>
using Vec4 = util::array<T, 4>;

template<int P>
struct TermSize : public stl::integral_constant<int, P * (P + 1) * (P + 2) / 6>
{
};

template<class T, int P>
using SphericalMultipole = util::array<T, TermSize<P>{}>;

template<int ArraySize, class = void>
struct ExpansionOrder
{
};

template<>
struct ExpansionOrder<TermSize<1>{}> : stl::integral_constant<int, 1>
{
};

template<>
struct ExpansionOrder<TermSize<2>{}> : stl::integral_constant<int, 2>
{
};

template<>
struct ExpansionOrder<TermSize<3>{}> : stl::integral_constant<int, 3>
{
};

template<>
struct ExpansionOrder<TermSize<4>{}> : stl::integral_constant<int, 4>
{
};

namespace ryoanji
{

struct GpuConfig
{
    //! @brief number of threads per warp
    #if defined(__CUDACC__) && !defined(__HIPCC__)
    static constexpr int warpSize = 32;
    #else
    static constexpr int warpSize = 64;
    #endif

    static_assert(warpSize == 32 || warpSize == 64, "warp size has to be 32 or 64");

    //! @brief log2(warpSize)
    static constexpr int warpSizeLog2 = (warpSize == 32) ? 5 : 6;

    /*! @brief integer type for representing a thread mask, e.g. return value of __ballot_sync()
     *
     * This will automatically pick the right type based on the warpSize choice. Do not adapt.
     */
    using ThreadMask = std::conditional_t<warpSize == 32, uint32_t, uint64_t>;

    static int getSmCount()
    {
        cudaDeviceProp prop;
        checkGpuErrors(cudaGetDeviceProperties(&prop, 0));
        return prop.multiProcessorCount;
    }

    //! @brief number of multiprocessors
    inline static int smCount = getSmCount();
};

//! Center and radius of bounding box
template<class T>
struct Box
{
    Vec3<T> X; //!< Box center
    T R; //!< Box radius
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

    HOST_DEVICE_FUN CellData(const unsigned int level, const unsigned int parent, const unsigned int body,
                                 const unsigned int nbody, const unsigned int child = 0, const unsigned int nchild = 1)
    {
        unsigned parentPack = parent | (level << LEVEL_SHIFT);
        unsigned childPack  = child | ((nchild - 1) << CHILD_SHIFT);
        data                = make_uint4(parentPack, childPack, body, nbody);
    }

    HOST_DEVICE_FUN CellData(uint4 data_)
        : data(data_)
    {
    }

    HOST_DEVICE_FUN int level() const { return data.x >> LEVEL_SHIFT; }
    HOST_DEVICE_FUN int parent() const { return data.x & LEVEL_MASK; }
    HOST_DEVICE_FUN int child() const { return data.y & CHILD_MASK; }
    HOST_DEVICE_FUN int nchild() const { return (data.y >> CHILD_SHIFT) + 1; }
    HOST_DEVICE_FUN int body() const { return data.z; }
    HOST_DEVICE_FUN int nbody() const { return data.w; }
    HOST_DEVICE_FUN bool isLeaf() const { return data.y == 0; }
    HOST_DEVICE_FUN bool isNode() const { return !isLeaf(); }

    HOST_DEVICE_FUN void setParent(unsigned parent) { data.x = parent | (level() << LEVEL_SHIFT); }
    HOST_DEVICE_FUN void setChild(unsigned child) { data.y = child | ((nchild() - 1) << CHILD_SHIFT); }
    HOST_DEVICE_FUN void setBody(unsigned body_) { data.z = body_; }
    HOST_DEVICE_FUN void setNBody(unsigned nbody_) { data.w = nbody_; }
};

template<class T>
inline __host__ __device__ Vec3<T> makeVec3(Vec4<T> v)
{
    Vec3<T> ret;
    ret[0] = v[0];
    ret[1] = v[1];
    ret[2] = v[2];
    return ret;
}

static void kernelSuccess(const char kernel[] = "kernel")
{
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "%s launch failed: %s\n", kernel, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

} // namespace ryoanji
