#pragma once

#include <cstdio>
#include <vector>
#include "BBox.hpp"
#include "sph/kernels.hpp"
#include "sph/lookupTables.hpp"


#if defined(USE_CUDA)
#include "LinearOctree.hpp"
#include "sph/cuda/cudaUtils.cuh"

namespace sphexa
{
template<typename T>
class DeviceLinearOctree
{
public:
    int size;
    int *ncells;
    int *cells;
    int *localPadding;
    int *localParticleCount;
    T *xmin, *xmax, *ymin, *ymax, *zmin, *zmax;
    T xmin0, xmax0, ymin0, ymax0, zmin0, zmax0;

    void mapLinearOctreeToDevice(const LinearOctree<T> &o)
    {
        size_t size_int = o.size * sizeof(int);
        size_t size_T = o.size * sizeof(T);

        size = o.size;

        CHECK_CUDA_ERR(sphexa::sph::cuda::utils::cudaMalloc(size_int * 8, cells));
        CHECK_CUDA_ERR(sphexa::sph::cuda::utils::cudaMalloc(size_int, ncells, localPadding, localParticleCount));
        CHECK_CUDA_ERR(sphexa::sph::cuda::utils::cudaMalloc(size_T, xmin, xmax, ymin, ymax, zmin, zmax));

        CHECK_CUDA_ERR(cudaMemcpy(cells, o.cells.data(), size_int * 8, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(ncells, o.ncells.data(), size_int, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(localPadding, o.localPadding.data(), size_int, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(localParticleCount, o.localParticleCount.data(), size_int, cudaMemcpyHostToDevice));

        CHECK_CUDA_ERR(cudaMemcpy(xmin, o.xmin.data(), size_T, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(xmax, o.xmax.data(), size_T, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(ymin, o.ymin.data(), size_T, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(ymax, o.ymax.data(), size_T, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(zmin, o.zmin.data(), size_T, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(zmax, o.zmax.data(), size_T, cudaMemcpyHostToDevice));

        xmin0 = o.xmin[0];
        xmax0 = o.xmax[0];
        ymin0 = o.ymin[0];
        ymax0 = o.ymax[0];
        zmin0 = o.zmin[0];
        zmax0 = o.zmax[0];
    }

    void unmapLinearOctreeFromDevice()
    {
        CHECK_CUDA_ERR(sphexa::sph::cuda::utils::cudaFree(cells));
        CHECK_CUDA_ERR(sphexa::sph::cuda::utils::cudaFree(ncells, localPadding, localParticleCount));
        CHECK_CUDA_ERR(sphexa::sph::cuda::utils::cudaFree(xmin, xmax, ymin, ymax, zmin, zmax));
    }
};
}

#endif

namespace sphexa
{
template <typename T>
struct ParticlesDataSqPatch
{
    inline void resize(const size_t size)
    {
        for (unsigned int i = 0; i < data.size(); ++i)
            data[i]->resize(size);
    }

    size_t iteration;                            // Current iteration
    size_t n, side, count;                       // Number of particles
    std::vector<T> x, y, z, x_m1, y_m1, z_m1;    // Positions
    std::vector<T> vx, vy, vz;                   // Velocities
    std::vector<T> ro;                           // Density
    std::vector<T> u;                            // Internal Energy
    std::vector<T> p;                            // Pressure
    std::vector<T> h;                            // Smoothing Length
    std::vector<T> m;                            // Mass
    std::vector<T> c;                            // Speed of sound
    std::vector<T> grad_P_x, grad_P_y, grad_P_z; // gradient of the pressure
    std::vector<T> du, du_m1;                    // variation of the energy
    std::vector<T> dt, dt_m1;
    std::vector<T> c11, c12, c13, c22, c23, c33; // IAD components
    std::vector<T> maxvsignal;

    // For Sedov
    std::vector<T> mui, temp, cv;

    T ttot, etot, ecin, eint;
    T minDt;

    BBox<T> bbox;

    std::vector<std::vector<T> *> data{&x,   &y,   &z,   &x_m1, &y_m1, &z_m1,     &vx,         &vy,       &vz,   &ro,    &u,
                                       &p,   &h,   &m,    &c,    &grad_P_x, &grad_P_y,   &grad_P_z, &du,   &du_m1, &dt,   &dt_m1,
                                       &c11, &c12, &c13, &c22,  &c23,  &c33,      &maxvsignal, &mui,      &temp, &cv};

    const std::array<double, lt::size> wh = lt::createWharmonicLookupTable<double, lt::size>();
    const std::array<double, lt::size> whd = lt::createWharmonicDerivativeLookupTable<double, lt::size>();

#ifdef USE_MPI
    MPI_Comm comm;
    int pnamelen = 0;
    char pname[MPI_MAX_PROCESSOR_NAME];
#endif

    int rank = 0;
    int nrank = 1;

    constexpr static T sincIndex = 6.0;
    constexpr static T Kcour = 0.2;
    constexpr static T maxDtIncrease = 1.1;
    const static T K;
};

template <typename T>
struct ParticlesDataSqPatch
{
    inline void resize(const size_t size)
    {
        for (unsigned int i = 0; i < data.size(); ++i)
            data[i]->resize(size);
    }

    size_t iteration;                            // Current iteration
    size_t n, side, count;                       // Number of particles
    std::vector<T> x, y, z, x_m1, y_m1, z_m1;    // Positions
    std::vector<T> vx, vy, vz;                   // Velocities
    std::vector<T> ro, ro_0;                     // Density
    std::vector<T> u;                            // Internal Energy
    std::vector<T> p, p_0;                            // Pressure
    std::vector<T> h;                            // Smoothing Length
    std::vector<T> m;                            // Mass
    std::vector<T> c;                            // Speed of sound
    std::vector<T> grad_P_x, grad_P_y, grad_P_z; // gradient of the pressure
    std::vector<T> du, du_m1;                    // variation of the energy
    std::vector<T> dt, dt_m1;
    std::vector<T> c11, c12, c13, c22, c23, c33; // IAD components
    std::vector<T> maxvsignal;

    // For Sedov
    std::vector<T> mui, temp, cv;

    T ttot, etot, ecin, eint;
    T minDt;

    BBox<T> bbox;

    std::vector<std::vector<T> *> data{&x,   &y,   &z,   &x_m1, &y_m1, &z_m1,     &vx,         &vy,       &vz,   &ro, &ro_0,   &u,
                                       &p,   &p_0, &h,   &m,    &c,    &grad_P_x, &grad_P_y,   &grad_P_z, &du,   &du_m1, &dt,   &dt_m1,
                                       &c11, &c12, &c13, &c22,  &c23,  &c33,      &maxvsignal, &mui,      &temp, &cv};

    const std::array<double, lt::size> wh = lt::createWharmonicLookupTable<double, lt::size>();
    const std::array<double, lt::size> whd = lt::createWharmonicDerivativeLookupTable<double, lt::size>();

#ifdef USE_MPI
    MPI_Comm comm;
    int pnamelen = 0;
    char pname[MPI_MAX_PROCESSOR_NAME];
#endif


#if defined(USE_CUDA)
    // number of CUDA streams to use
    static const int NST = 2;

    cudaStream_t streams[NST];
    
    int *d_clist[NST], *d_neighbors[NST], *d_neighborsCount[NST]; // work arrays per stream
    T *d_x, *d_y, *d_z, *d_vx, *d_vy, *d_vz, *d_m, *d_h, *d_ro, *d_p, *d_c, *d_c11, *d_c12, *d_c13, *d_c22, *d_c23, *d_c33, *d_wh, *d_whd;
    BBox<T> *d_bbox;
    T *d_grad_P_x, *d_grad_P_y, *d_grad_P_z, *d_du, *d_maxvsignal;

    DeviceLinearOctree<T> d_o;

    void moveParticleDataToDevice(const LinearOctree<T> &o, const std::vector<Task> &taskList)
    {
        const size_t np = x.size();
        const size_t size_np_T = np * sizeof(T);
        const T ngmax = taskList.empty() ? 0 : taskList.front().ngmax;

        const size_t ltsize = wh.size();
        const size_t size_lt_T = ltsize * sizeof(T);
    
        const auto largestChunkSize =
            std::max_element(taskList.cbegin(), taskList.cend(),
                             [](const Task &lhs, const Task &rhs) { return lhs.clist.size() < rhs.clist.size(); })
                ->clist.size();

        const size_t size_largerNeighborsChunk_int = largestChunkSize * ngmax * sizeof(int);
        const size_t size_largerNChunk_int = largestChunkSize * sizeof(int);
        const size_t size_bbox = sizeof(BBox<T>);

        // initialize streams
        for (int i = 0; i < NST; ++i)
            CHECK_CUDA_ERR(cudaStreamCreate(&streams[i]));

        CHECK_CUDA_ERR(sph::cuda::utils::cudaMalloc(size_np_T, d_x, d_y, d_z, d_vx, d_vy, d_vz, d_h, d_m, d_ro, d_p, d_c, d_c11, d_c12, d_c13, d_c22, d_c23, d_c33, d_grad_P_x, d_grad_P_y, d_grad_P_z, d_du, d_maxvsignal));
        CHECK_CUDA_ERR(sph::cuda::utils::cudaMalloc(size_lt_T, d_wh, d_whd));
        CHECK_CUDA_ERR(sph::cuda::utils::cudaMalloc(size_bbox, d_bbox));

        CHECK_CUDA_ERR(cudaMemcpy(d_x, x.data(), size_np_T, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(d_y, y.data(), size_np_T, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(d_z, z.data(), size_np_T, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(d_vx, vx.data(), size_np_T, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(d_vy, vy.data(), size_np_T, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(d_vz, vz.data(), size_np_T, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(d_h, h.data(), size_np_T, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(d_m, m.data(), size_np_T, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(d_p, p.data(), size_np_T, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(d_c, c.data(), size_np_T, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(d_wh, wh.data(), size_lt_T, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(d_whd, whd.data(), size_lt_T, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(d_bbox, &bbox, size_bbox, cudaMemcpyHostToDevice));

        CHECK_CUDA_ERR(sph::cuda::utils::cudaMalloc(size_bbox, d_bbox));
        for (int i = 0; i < NST; ++i)
            CHECK_CUDA_ERR(sph::cuda::utils::cudaMalloc(size_largerNChunk_int, d_clist[i], d_neighborsCount[i]));
        for (int i = 0; i < NST; ++i)
            CHECK_CUDA_ERR(sph::cuda::utils::cudaMalloc(size_largerNeighborsChunk_int, d_neighbors[i]));

        d_o.mapLinearOctreeToDevice(o);
    }

    void freeDeviceParticleData()
    {
        for (int i = 0; i < NST; ++i)
        CHECK_CUDA_ERR(cudaStreamDestroy(streams[i]));

        CHECK_CUDA_ERR(sph::cuda::utils::cudaFree(d_bbox, d_x, d_y, d_z, d_vx, d_vy, d_vz, d_h, d_m, d_ro, d_p,
            d_c, d_c11, d_c12, d_c13, d_c22, d_c23, d_c33, d_grad_P_x, d_grad_P_y, d_grad_P_z, d_du, d_maxvsignal, d_wh, d_whd));


        for (int i = 0; i < NST; ++i)
            CHECK_CUDA_ERR(sph::cuda::utils::cudaFree(d_clist[i], d_neighbors[i], d_neighborsCount[i]));
        
        d_o.unmapLinearOctreeFromDevice();
    }
#endif


    int rank = 0;
    int nrank = 1;

    constexpr static T sincIndex = 6.0;
    constexpr static T Kcour = 0.2;
    constexpr static T maxDtIncrease = 1.1;
    const static T K;
};

template <typename T>
const T ParticlesData<T>::K = sphexa::compute_3d_k(sincIndex);

template <typename T>
struct ParticlesDataEvrard
{
    inline void resize(const size_t size)
    {
        for (unsigned int i = 0; i < data.size(); ++i)
            data[i]->resize(size);
    }

    size_t iteration;                            // Current iteration
    size_t n, side, count;                       // Number of particles
    std::vector<T> x, y, z, x_m1, y_m1, z_m1;    // Positions
    std::vector<T> vx, vy, vz;                   // Velocities
    std::vector<T> ro, ro_0;                     // Density
    std::vector<T> u;                            // Internal Energy
    std::vector<T> p, p_0;                       // Pressure
    std::vector<T> h;                            // Smoothing Length
    std::vector<T> m;                            // Mass
    std::vector<T> c;                            // Speed of sound
    std::vector<T> grad_P_x, grad_P_y, grad_P_z; // gradient of the pressure
    std::vector<T> du, du_m1;                    // variation of the energy
    std::vector<T> dt, dt_m1;
    std::vector<T> c11, c12, c13, c22, c23, c33; // IAD components
    std::vector<T> maxvsignal;

    std::vector<T> fx, fy, fz, ugrav; // Gravity
    std::vector<T> cv;                // Specific heat
    std::vector<T> temp;              // Temperature
    std::vector<T> mue;               // Mean molecular weigh of electrons
    std::vector<T> mui;               // Mean molecular weight of ions

    T ttot = 0.0, etot, ecin, eint, egrav = 0.0;
    T minDt, minDmy = 1e-4, minTmpDt;

    sphexa::BBox<T> bbox;

    std::vector<std::vector<T> *> data{&x,   &y,   &z,   &x_m1, &y_m1,       &z_m1,     &vx,       &vy, &vz,    &ro, &ro_0,  &u,   &p,
                                       &p_0, &h,   &m,   &c,    &grad_P_x,   &grad_P_y, &grad_P_z, &du, &du_m1, &dt, &dt_m1, &c11, &c12,
                                       &c13, &c22, &c23, &c33,  &maxvsignal, &fx,       &fy,       &fz, &ugrav, &cv, &temp,  &mue, &mui};


#ifdef USE_MPI
    MPI_Comm comm;
    int pnamelen = 0;
    char pname[MPI_MAX_PROCESSOR_NAME];
#endif

    int rank = 0;
    int nrank = 1;

    constexpr static T g = 1.0; // for Evrard Collapse Gravity.
    // constexpr static T g = 6.6726e-8; // the REAL value of g. g is 1.0 for Evrard mainly

    constexpr static T sincIndex = 5.0;
    constexpr static T Kcour = 0.2;
    constexpr static T maxDtIncrease = 1.1;
    constexpr static size_t ngmin = 5, ng0 = 100, ngmax = 150;
    const static T K;
};

template <typename T>
const T ParticlesDataEvrard<T>::K = sphexa::compute_3d_k(sincIndex);

} // namespace sphexa
