#include <algorithm>

#include "sph.cuh"
#include "BBox.hpp"
#include "ParticlesData.hpp"
#include "cudaUtils.cuh"
#include "../kernel/computeMomentumAndEnergy.hpp"

#include "cstone/cuda/findneighbors.cuh"

namespace sphexa
{
namespace sph
{
namespace cuda
{

template<class T>
__global__ void computeMomentumAndEnergyIAD(int n, T sincIndex, T K, int ngmax, BBox<T> bbox,
                                            const int* clist, const int* neighbors, const int* neighborsCount,
                                            const T* x, const T* y, const T* z, const T* vx, const T* vy, const T* vz,
                                            const T* h, const T* m, const T* ro, const T* p, const T* c,
                                            const T* c11, const T* c12, const T* c13, const T* c22, const T* c23,
                                            const T* c33, const T* wh, const T* whd,
                                            T* grad_P_x, T* grad_P_y, T* grad_P_z, T* du, T* maxvsignal)
{
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= n) return;

    sph::kernels::momentumAndEnergyJLoop(tid, sincIndex, K, ngmax, bbox, clist, neighbors, neighborsCount,
                                         x, y, z, vx, vy, vz, h, m, ro, p, c, c11, c12, c13, c22, c23, c33,
                                         wh, whd, grad_P_x, grad_P_y, grad_P_z, du, maxvsignal);
}

template <class Dataset>
void computeMomentumAndEnergyIAD(const std::vector<Task> &taskList, Dataset &d, const cstone::Box<double>& box)
{
    using T = typename Dataset::RealType;

    size_t np = d.x.size();
    size_t size_np_T = np * sizeof(T);
    T ngmax = taskList.empty() ? 0 : taskList.front().ngmax;

    auto largestChunkSize =
        std::max_element(taskList.cbegin(), taskList.cend(),
                         [](const Task &lhs, const Task &rhs) { return lhs.clist.size() < rhs.clist.size(); })
            ->clist.size();

    d.devPtrs.resize_streams(largestChunkSize, ngmax);

    // number of CUDA streams to use
    constexpr int NST = DeviceParticlesData<T, Dataset>::NST;

    size_t ltsize = d.wh.size();

    BBox<T> bbox{
        box.xmin(), box.xmax(), box.ymin(), box.ymax(), box.zmin(), box.zmax(), box.pbcX(), box.pbcY(), box.pbcZ()};

    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_vx, d.vx.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_vy, d.vy.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_vz, d.vz.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_p, d.p.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_c, d.c.data(), size_np_T, cudaMemcpyHostToDevice));

    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_c11, d.c11.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_c12, d.c12.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_c13, d.c13.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_c22, d.c22.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_c23, d.c23.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_c33, d.c33.data(), size_np_T, cudaMemcpyHostToDevice));

    for (int i = 0; i < taskList.size(); ++i)
    {
        const auto& t = taskList[i];

        int sIdx = i % NST;
        cudaStream_t stream = d.devPtrs.d_stream[sIdx].stream;

        int* d_clist_use = d.devPtrs.d_stream[sIdx].d_clist;
        int* d_neighbors_use = d.devPtrs.d_stream[sIdx].d_neighbors;
        int* d_neighborsCount_use = d.devPtrs.d_stream[sIdx].d_neighborsCount;

        size_t n = t.clist.size();
        size_t size_n_int = n * sizeof(int);

        CHECK_CUDA_ERR(cudaMemcpyAsync(d_clist_use, t.clist.data(), size_n_int, cudaMemcpyHostToDevice, stream));

        findNeighborsHilbertGpu(d.devPtrs.d_x, d.devPtrs.d_y, d.devPtrs.d_z, d.devPtrs.d_h,
                                t.clist[0], t.clist[n - 1] + 1, np, box, d.devPtrs.d_codes,
                                d_neighbors_use, d_neighborsCount_use, ngmax, stream);
        CHECK_CUDA_ERR(cudaGetLastError());

        unsigned numThreads = 256;
        unsigned numBlocks  = (n + numThreads - 1) / numThreads;

        computeMomentumAndEnergyIAD<<<numBlocks, numThreads, 0, stream>>>(
            n, d.sincIndex, d.K, ngmax, bbox, d_clist_use, d_neighbors_use, d_neighborsCount_use,
            d.devPtrs.d_x, d.devPtrs.d_y, d.devPtrs.d_z, d.devPtrs.d_vx, d.devPtrs.d_vy, d.devPtrs.d_vz,
            d.devPtrs.d_h, d.devPtrs.d_m, d.devPtrs.d_ro, d.devPtrs.d_p, d.devPtrs.d_c,
            d.devPtrs.d_c11, d.devPtrs.d_c12, d.devPtrs.d_c13, d.devPtrs.d_c22, d.devPtrs.d_c23, d.devPtrs.d_c33,
            d.devPtrs.d_wh, d.devPtrs.d_whd, d.devPtrs.d_grad_P_x, d.devPtrs.d_grad_P_y, d.devPtrs.d_grad_P_z,
            d.devPtrs.d_du, d.devPtrs.d_maxvsignal);

        CHECK_CUDA_ERR(cudaGetLastError());
    }

    CHECK_CUDA_ERR(cudaMemcpy(d.grad_P_x.data(), d.devPtrs.d_grad_P_x, size_np_T, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(d.grad_P_y.data(), d.devPtrs.d_grad_P_y, size_np_T, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(d.grad_P_z.data(), d.devPtrs.d_grad_P_z, size_np_T, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(d.du.data(), d.devPtrs.d_du, size_np_T, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(d.maxvsignal.data(), d.devPtrs.d_maxvsignal, size_np_T, cudaMemcpyDeviceToHost));
}

template void computeMomentumAndEnergyIAD(const std::vector<Task>& taskList, ParticlesData<double, unsigned>& d,
                                          const cstone::Box<double>&);
template void computeMomentumAndEnergyIAD(const std::vector<Task>& taskList, ParticlesData<double, uint64_t>& d,
                                          const cstone::Box<double>&);

} // namespace cuda
} // namespace sph
} // namespace sphexa
