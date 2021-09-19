#pragma once

#include "LinearOctree.hpp"
#include "cudaUtils.cuh"
#include "BBox.hpp"

namespace sphexa
{
namespace sph
{
namespace cuda
{

template <typename T>
struct DeviceLinearOctree
{
    int size;
    int *ncells;
    int *cells;
    int *localPadding;
    int *localParticleCount;
    T *xmin, *xmax, *ymin, *ymax, *zmin, *zmax;
    T xmin0, xmax0, ymin0, ymax0, zmin0, zmax0;

    bool already_mapped = false;

    void mapLinearOctreeToDevice(const LinearOctree<T> &o)
    {
        if (already_mapped)
        {
            unmapLinearOctreeFromDevice();
        }
        size_t size_int = o.size * sizeof(int);
        size_t size_T = o.size * sizeof(T);

        size = o.size;

        CHECK_CUDA_ERR(sphexa::sph::cuda::utils::hipMalloc(size_int * 8, cells));
        CHECK_CUDA_ERR(sphexa::sph::cuda::utils::hipMalloc(size_int, ncells, localPadding, localParticleCount));
        CHECK_CUDA_ERR(sphexa::sph::cuda::utils::hipMalloc(size_T, xmin, xmax, ymin, ymax, zmin, zmax));

        CHECK_CUDA_ERR(hipMemcpy(cells, o.cells.data(), size_int * 8, hipMemcpyHostToDevice));
        CHECK_CUDA_ERR(hipMemcpy(ncells, o.ncells.data(), size_int, hipMemcpyHostToDevice));
        CHECK_CUDA_ERR(hipMemcpy(localPadding, o.localPadding.data(), size_int, hipMemcpyHostToDevice));
        CHECK_CUDA_ERR(hipMemcpy(localParticleCount, o.localParticleCount.data(), size_int, hipMemcpyHostToDevice));

        CHECK_CUDA_ERR(hipMemcpy(xmin, o.xmin.data(), size_T, hipMemcpyHostToDevice));
        CHECK_CUDA_ERR(hipMemcpy(xmax, o.xmax.data(), size_T, hipMemcpyHostToDevice));
        CHECK_CUDA_ERR(hipMemcpy(ymin, o.ymin.data(), size_T, hipMemcpyHostToDevice));
        CHECK_CUDA_ERR(hipMemcpy(ymax, o.ymax.data(), size_T, hipMemcpyHostToDevice));
        CHECK_CUDA_ERR(hipMemcpy(zmin, o.zmin.data(), size_T, hipMemcpyHostToDevice));
        CHECK_CUDA_ERR(hipMemcpy(zmax, o.zmax.data(), size_T, hipMemcpyHostToDevice));

        xmin0 = o.xmin[0];
        xmax0 = o.xmax[0];
        ymin0 = o.ymin[0];
        ymax0 = o.ymax[0];
        zmin0 = o.zmin[0];
        zmax0 = o.zmax[0];
        already_mapped = true;
    }

    void unmapLinearOctreeFromDevice()
    {
        CHECK_CUDA_ERR(sphexa::sph::cuda::utils::hipFree(cells));
        CHECK_CUDA_ERR(sphexa::sph::cuda::utils::hipFree(ncells, localPadding, localParticleCount));
        CHECK_CUDA_ERR(sphexa::sph::cuda::utils::hipFree(xmin, xmax, ymin, ymax, zmin, zmax));
    }
};

template <typename T, class ParticleData>
class DeviceParticlesData
{
    size_t allocatedDeviceMemory = 0;
    size_t allocatedTaskSize = 0;

public:
    // number of CUDA streams to use
    static constexpr int NST = 2;

    struct neighbors_stream
    {
        hipStream_t stream;
        int *d_clist, *d_neighbors, *d_neighborsCount;
    };

    struct neighbors_stream d_stream[NST];

    T *d_x, *d_y, *d_z, *d_vx, *d_vy, *d_vz, *d_m, *d_h, *d_ro, *d_p, *d_c,
      *d_c11, *d_c12, *d_c13, *d_c22, *d_c23, *d_c33, *d_wh, *d_whd,
      *d_grad_P_x, *d_grad_P_y, *d_grad_P_z, *d_du, *d_maxvsignal;

    BBox<T> *d_bbox;

    typename ParticleData::KeyType *d_codes;

    [[nodiscard]] size_t capacity() const { return allocatedDeviceMemory; }

    void resize(size_t size)
    {
        if (size > allocatedDeviceMemory)
        {
            // TODO: Investigate benefits of low-level reallocate
            if (allocatedDeviceMemory)
            {
                CHECK_CUDA_ERR(utils::hipFree(d_x, d_y, d_z, d_h, d_m, d_ro));
                CHECK_CUDA_ERR(utils::hipFree(d_c11, d_c12, d_c13, d_c22, d_c23, d_c33));
                CHECK_CUDA_ERR(utils::hipFree(d_vx, d_vy, d_vz, d_p, d_c, d_grad_P_x, d_grad_P_y, d_grad_P_z, d_du,
                                               d_maxvsignal));

                CHECK_CUDA_ERR(utils::hipFree(d_codes));
            }

            size = size_t(double(size) * 1.01); // allocate 1% extra to avoid reallocation on small size increase

            size_t size_np_T       = size * sizeof(T);
            size_t size_np_KeyType = size * sizeof(typename ParticleData::KeyType);

            CHECK_CUDA_ERR(utils::hipMalloc(size_np_T, d_x, d_y, d_z, d_h, d_m, d_ro));
            CHECK_CUDA_ERR(utils::hipMalloc(size_np_T, d_c11, d_c12, d_c13, d_c22, d_c23, d_c33));
            CHECK_CUDA_ERR(utils::hipMalloc(size_np_T, d_vx, d_vy, d_vz, d_p, d_c, d_grad_P_x, d_grad_P_y,
                                             d_grad_P_z, d_du, d_maxvsignal));
            CHECK_CUDA_ERR(utils::hipMalloc(size_np_KeyType, d_codes));

            allocatedDeviceMemory = size;
        }
    }

    void resize_streams(size_t taskSize, size_t ngmax)
    {
        if (taskSize > allocatedTaskSize)
        {
            if (allocatedTaskSize)
            {
                //printf("[D] increased stream size from %ld to %ld\n", allocatedTaskSize, taskSize);
                for (int i = 0; i < NST; ++i)
                {
                    CHECK_CUDA_ERR(utils::hipFree(d_stream[i].d_clist, d_stream[i].d_neighborsCount));
                    CHECK_CUDA_ERR(utils::hipFree(d_stream[i].d_neighbors));
                }
            }

            taskSize = size_t(double(taskSize) * 1.01); // allocate 1% extra to avoid reallocation on small size increase

            for (int i = 0; i < NST; ++i)
            {
                CHECK_CUDA_ERR(
                    utils::hipMalloc(taskSize * sizeof(int), d_stream[i].d_clist, d_stream[i].d_neighborsCount));
                CHECK_CUDA_ERR(utils::hipMalloc(taskSize * ngmax * sizeof(int), d_stream[i].d_neighbors));
            }

            allocatedTaskSize = taskSize;
        }
    }

    DeviceParticlesData() = delete;

    explicit DeviceParticlesData(const ParticleData& pd)
    {
        const size_t size_bbox = sizeof(BBox<T>);

        const size_t ltsize = pd.wh.size();
        const size_t size_lt_T = ltsize * sizeof(T);

        CHECK_CUDA_ERR(utils::hipMalloc(size_lt_T, d_wh, d_whd));
        CHECK_CUDA_ERR(utils::hipMalloc(size_bbox, d_bbox));

        for (int i = 0; i < NST; ++i)
        {
            CHECK_CUDA_ERR(hipStreamCreate(&d_stream[i].stream));
        }
    }

    ~DeviceParticlesData()
    {
        CHECK_CUDA_ERR(utils::hipFree(d_bbox, d_x, d_y, d_z, d_vx, d_vy, d_vz, d_h, d_m, d_ro, d_p, d_c,
                                       d_c11, d_c12, d_c13, d_c22, d_c23, d_c33, d_grad_P_x, d_grad_P_y, d_grad_P_z,
                                       d_du, d_maxvsignal, d_wh, d_whd));
        CHECK_CUDA_ERR(utils::hipFree(d_codes));

        for (int i = 0; i < NST; ++i)
        {
            CHECK_CUDA_ERR(hipStreamDestroy(d_stream[i].stream));
        }

        for (int i = 0; i < NST; ++i)
        {
            CHECK_CUDA_ERR(utils::hipFree(d_stream[i].d_clist, d_stream[i].d_neighbors, d_stream[i].d_neighborsCount));
        }
    }
};
} // namespace cuda
} // namespace sph
} // namespace sphexa
