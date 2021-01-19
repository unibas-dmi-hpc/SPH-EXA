#include <cuda.h>
#include <algorithm>

#include "sph.cuh"
#include "BBox.hpp"
#include "ParticlesData.hpp"
#include "cudaUtils.cuh"
#include "../kernels.hpp"
#include "../lookupTables.hpp"

namespace sphexa
{
namespace sph
{
namespace cuda
{
namespace kernels
{
template <typename T>
__global__ void computeIAD(const int n, const T sincIndex, const T K, const int ngmax, const BBox<T> *bbox, const int *clist,
                           const int *neighbors, const int *neighborsCount, const T *x, const T *y, const T *z, const T *h, const T *m, 
                           const T *ro, const T *wh, const T *whd, const size_t ltsize, T *c11, T *c12, T *c13, T *c22, T *c23, T *c33)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= n) return;

    const int i = clist[tid];
    const int nn = neighborsCount[tid];

    T tau11 = 0.0, tau12 = 0.0, tau13 = 0.0, tau22 = 0.0, tau23 = 0.0, tau33 = 0.0;
    for (int pj = 0; pj < nn; ++pj)
    {
        const int j = neighbors[tid * ngmax + pj];

        const T dist = distancePBC(*bbox, h[i], x[i], y[i], z[i], x[j], y[j], z[j]);
        const T vloc = dist / h[i];

        const T w = K * math_namespace::pow(lt::wharmonic_lt_with_derivative(wh, whd, ltsize, vloc), (int)sincIndex);
        const T W = w / (h[i] * h[i] * h[i]);

        T r_ijx = (x[i] - x[j]);
        T r_ijy = (y[i] - y[j]);
        T r_ijz = (z[i] - z[j]);

        applyPBC(*bbox, 2.0 * h[i], r_ijx, r_ijy, r_ijz);

        tau11 += r_ijx * r_ijx * m[j] / ro[j] * W;
        tau12 += r_ijx * r_ijy * m[j] / ro[j] * W;
        tau13 += r_ijx * r_ijz * m[j] / ro[j] * W;
        tau22 += r_ijy * r_ijy * m[j] / ro[j] * W;
        tau23 += r_ijy * r_ijz * m[j] / ro[j] * W;
        tau33 += r_ijz * r_ijz * m[j] / ro[j] * W;
    }

    const T det =
        tau11 * tau22 * tau33 + 2.0 * tau12 * tau23 * tau13 - tau11 * tau23 * tau23 - tau22 * tau13 * tau13 - tau33 * tau12 * tau12;

    c11[i] = (tau22 * tau33 - tau23 * tau23) / det;
    c12[i] = (tau13 * tau23 - tau33 * tau12) / det;
    c13[i] = (tau12 * tau23 - tau22 * tau13) / det;
    c22[i] = (tau11 * tau33 - tau13 * tau13) / det;
    c23[i] = (tau13 * tau12 - tau11 * tau23) / det;
    c33[i] = (tau11 * tau22 - tau12 * tau12) / det;
}
} // namespace kernels

template <typename T, class Dataset>
void computeIAD(const std::vector<Task> &taskList, Dataset &d)
{
    const int maz = d.bbox.PBCz ? 2 : 0;
    const int may = d.bbox.PBCy ? 2 : 0;
    const int max = d.bbox.PBCx ? 2 : 0;

    const T displx = d.devPtrs.d_o.xmax0 - d.devPtrs.d_o.xmin0;
    const T disply = d.devPtrs.d_o.ymax0 - d.devPtrs.d_o.ymin0;
    const T displz = d.devPtrs.d_o.zmax0 - d.devPtrs.d_o.zmin0;

    const size_t np = d.x.size();
    const size_t size_np_T = np * sizeof(T);
    const T ngmax = taskList.empty() ? 0 : taskList.front().ngmax;

    const size_t ltsize = d.wh.size();

    const auto largestChunkSize =
        std::max_element(taskList.cbegin(), taskList.cend(),
                         [](const Task &lhs, const Task &rhs) { return lhs.clist.size() < rhs.clist.size(); })
            ->clist.size();

    d.devPtrs.resize_streams(largestChunkSize, ngmax);

    // number of CUDA streams to use
    const int NST = DeviceParticlesData<T, Dataset>::NST;

    /*
    // device pointers - d_ prefix stands for device
    int *d_clist[NST], *d_neighbors[NST], *d_neighborsCount[NST]; // work arrays per stream

    // input data
    //CHECK_CUDA_ERR(utils::cudaMalloc(size_np_T, d.d_c11, d.d_c12, d.d_c13, d.d_c22, d.d_c23, d.d_c33));

    for (int i = 0; i < NST; ++i)
        CHECK_CUDA_ERR(utils::cudaMalloc(size_largerNChunk_int, d_clist[i], d_neighborsCount[i]));
    for (int i = 0; i < NST; ++i)
        CHECK_CUDA_ERR(utils::cudaMalloc(size_largerNeighborsChunk_int, d_neighbors[i]));

    cudaStream_t streams[NST];
    for (int i = 0; i < NST; ++i)
        CHECK_CUDA_ERR(cudaStreamCreate(&streams[i]));
    */
    
    // CHECK_CUDA_ERR(cudaMemcpy(d_x, d.x.data(), size_np_T, cudaMemcpyHostToDevice));
    // CHECK_CUDA_ERR(cudaMemcpy(d_y, d.y.data(), size_np_T, cudaMemcpyHostToDevice));
    // CHECK_CUDA_ERR(cudaMemcpy(d_z, d.z.data(), size_np_T, cudaMemcpyHostToDevice));
    // CHECK_CUDA_ERR(cudaMemcpy(d_h, d.h.data(), size_np_T, cudaMemcpyHostToDevice));
    // CHECK_CUDA_ERR(cudaMemcpy(d_m, d.m.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_ro, d.ro.data(), size_np_T, cudaMemcpyHostToDevice));
    // CHECK_CUDA_ERR(cudaMemcpy(d_wh, d.wh.data(), size_lt_T, cudaMemcpyHostToDevice));
    // CHECK_CUDA_ERR(cudaMemcpy(d_whd, d.whd.data(), size_lt_T, cudaMemcpyHostToDevice));
    // CHECK_CUDA_ERR(cudaMemcpy(d_bbox, &d.bbox, size_bbox, cudaMemcpyHostToDevice));

    //DeviceLinearOctree<T> d_o;
    //d.d_o.mapLinearOctreeToDevice(o);
    
    for (int i = 0; i < taskList.size(); ++i)
    {
        const auto &t = taskList[i];

        const int sIdx = i % NST;
        /*
        cudaStream_t stream = streams[sIdx];

        int *d_clist_use = d_clist[sIdx];
        int *d_neighbors_use = d_neighbors[sIdx];
        int *d_neighborsCount_use = d_neighborsCount[sIdx];
        */
        cudaStream_t stream = d.devPtrs.d_stream[sIdx].stream;

        int *d_clist_use = d.devPtrs.d_stream[sIdx].d_clist;
        int *d_neighbors_use = d.devPtrs.d_stream[sIdx].d_neighbors;
        int *d_neighborsCount_use = d.devPtrs.d_stream[sIdx].d_neighborsCount;

        const size_t n = t.clist.size();
        const size_t size_n_int = n * sizeof(int);
        // const size_t size_nNeighbors = n * ngmax * sizeof(int);

        CHECK_CUDA_ERR(cudaMemcpyAsync(d_clist_use, t.clist.data(), size_n_int, cudaMemcpyHostToDevice, stream));
        //CHECK_CUDA_ERR(cudaMemcpyAsync(d_neighbors_use, t.neighbors.data(), size_nNeighbors, cudaMemcpyHostToDevice, stream));
        //CHECK_CUDA_ERR(cudaMemcpyAsync(d_neighborsCount_use, t.neighborsCount.data(), size_n_int, cudaMemcpyHostToDevice, stream));

        const int threadsPerBlock = 256;
        const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

        kernels::findNeighbors<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
            d.devPtrs.d_o, d_clist_use, n, d.devPtrs.d_x, d.devPtrs.d_y, d.devPtrs.d_z, d.devPtrs.d_h, displx, disply, displz, max, may, maz, ngmax, d_neighbors_use, d_neighborsCount_use
        );

        // printf("CUDA IAD kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

        kernels::computeIAD<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(n, d.sincIndex, d.K, ngmax, d.devPtrs.d_bbox, d_clist_use, d_neighbors_use,
            d_neighborsCount_use, d.devPtrs.d_x, d.devPtrs.d_y, d.devPtrs.d_z, d.devPtrs.d_h, d.devPtrs.d_m, d.devPtrs.d_ro, d.devPtrs.d_wh, d.devPtrs.d_whd, ltsize, d.devPtrs.d_c11, d.devPtrs.d_c12, d.devPtrs.d_c13, d.devPtrs.d_c22,
            d.devPtrs.d_c23, d.devPtrs.d_c33);
        CHECK_CUDA_ERR(cudaGetLastError());
    }

    
    // d.d_o.unmapLinearOctreeFromDevice();

    CHECK_CUDA_ERR(cudaMemcpy(d.c11.data(), d.devPtrs.d_c11, size_np_T, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(d.c12.data(), d.devPtrs.d_c12, size_np_T, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(d.c13.data(), d.devPtrs.d_c13, size_np_T, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(d.c22.data(), d.devPtrs.d_c22, size_np_T, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(d.c23.data(), d.devPtrs.d_c23, size_np_T, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(d.c33.data(), d.devPtrs.d_c33, size_np_T, cudaMemcpyDeviceToHost));
    
    /*
    for (int i = 0; i < NST; ++i)
        CHECK_CUDA_ERR(cudaStreamDestroy(streams[i]));

    for (int i = 0; i < NST; ++i)
        CHECK_CUDA_ERR(utils::cudaFree(d_clist[i], d_neighbors[i], d_neighborsCount[i]));
    */
}

template void computeIAD<double, ParticlesData<double>>(const std::vector<Task> &taskList, ParticlesData<double> &d);

} // namespace cuda
} // namespace sph
} // namespace sphexa
