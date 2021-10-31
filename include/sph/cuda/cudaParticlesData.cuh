#pragma once

#include "cudaUtils.cuh"

namespace sphexa
{
namespace sph
{
namespace cuda
{

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
        cudaStream_t stream;
        int *d_neighborsCount;
    };

    struct neighbors_stream d_stream[NST];

    T *d_x, *d_y, *d_z, *d_vx, *d_vy, *d_vz, *d_m, *d_h, *d_ro, *d_p, *d_c,
      *d_c11, *d_c12, *d_c13, *d_c22, *d_c23, *d_c33, *d_wh, *d_whd,
      *d_grad_P_x, *d_grad_P_y, *d_grad_P_z, *d_du, *d_maxvsignal;

    typename ParticleData::KeyType *d_codes;

    [[nodiscard]] size_t capacity() const { return allocatedDeviceMemory; }

    void resize(size_t size)
    {
        if (size > allocatedDeviceMemory)
        {
            // TODO: Investigate benefits of low-level reallocate
            if (allocatedDeviceMemory)
            {
                CHECK_CUDA_ERR(utils::cudaFree(d_x, d_y, d_z, d_h, d_m, d_ro));
                CHECK_CUDA_ERR(utils::cudaFree(d_c11, d_c12, d_c13, d_c22, d_c23, d_c33));
                CHECK_CUDA_ERR(utils::cudaFree(d_vx, d_vy, d_vz, d_p, d_c, d_grad_P_x, d_grad_P_y, d_grad_P_z, d_du,
                                               d_maxvsignal));

                CHECK_CUDA_ERR(utils::cudaFree(d_codes));
            }

            size = size_t(double(size) * 1.01); // allocate 1% extra to avoid reallocation on small size increase

            size_t size_np_T       = size * sizeof(T);
            size_t size_np_KeyType = size * sizeof(typename ParticleData::KeyType);

            CHECK_CUDA_ERR(utils::cudaMalloc(size_np_T, d_x, d_y, d_z, d_h, d_m, d_ro));
            CHECK_CUDA_ERR(utils::cudaMalloc(size_np_T, d_c11, d_c12, d_c13, d_c22, d_c23, d_c33));
            CHECK_CUDA_ERR(utils::cudaMalloc(size_np_T, d_vx, d_vy, d_vz, d_p, d_c, d_grad_P_x, d_grad_P_y,
                                             d_grad_P_z, d_du, d_maxvsignal));
            CHECK_CUDA_ERR(utils::cudaMalloc(size_np_KeyType, d_codes));

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
                    CHECK_CUDA_ERR(utils::cudaFree(d_stream[i].d_neighborsCount));
                }
            }

            taskSize = size_t(double(taskSize) * 1.01); // allocate 1% extra to avoid reallocation on small size increase

            for (int i = 0; i < NST; ++i)
            {
                CHECK_CUDA_ERR(utils::cudaMalloc(taskSize * sizeof(int), d_stream[i].d_neighborsCount));
            }

            allocatedTaskSize = taskSize;
        }
    }

    DeviceParticlesData() = delete;

    explicit DeviceParticlesData(const ParticleData& pd)
    {
        size_t ltsize = pd.wh.size();
        size_t size_lt_T = ltsize * sizeof(T);

        CHECK_CUDA_ERR(utils::cudaMalloc(size_lt_T, d_wh, d_whd));

        for (int i = 0; i < NST; ++i)
        {
            CHECK_CUDA_ERR(cudaStreamCreate(&d_stream[i].stream));
        }
    }

    ~DeviceParticlesData()
    {
        CHECK_CUDA_ERR(utils::cudaFree(d_x, d_y, d_z, d_vx, d_vy, d_vz, d_h, d_m, d_ro, d_p, d_c,
                                       d_c11, d_c12, d_c13, d_c22, d_c23, d_c33, d_grad_P_x, d_grad_P_y, d_grad_P_z,
                                       d_du, d_maxvsignal, d_wh, d_whd));
        CHECK_CUDA_ERR(utils::cudaFree(d_codes));

        for (int i = 0; i < NST; ++i)
        {
            CHECK_CUDA_ERR(cudaStreamDestroy(d_stream[i].stream));
            CHECK_CUDA_ERR(utils::cudaFree(d_stream[i].d_neighborsCount));
        }
    }
};
} // namespace cuda
} // namespace sph
} // namespace sphexa
