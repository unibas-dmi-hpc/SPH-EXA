#pragma once

#include <cstdio>
#include <vector>
#include "sph/kernels.hpp"
#include "sph/lookupTables.hpp"

#if defined(USE_CUDA)
#include "sph/cuda/cudaParticlesData.cuh"
#endif

namespace sphexa
{
template<typename T, typename I>
struct ParticlesData
{
    using RealType = T;
    using KeyType  = I;

    inline void resize(size_t size)
    {
        size_t current_capacity = data[0]->capacity();
        if (size > current_capacity)
        {
            // limit reallocation growth to 5% instead of 200%
            size_t reserve_size = double(size) * 1.05;
            for (unsigned int i = 0; i < data.size(); ++i)
            {
                data[i]->reserve(reserve_size);
            }
            codes.reserve(reserve_size);
        }

        for (unsigned int i = 0; i < data.size(); ++i)
        {
            data[i]->resize(size);
        }

        codes.resize(size);

#if defined(USE_CUDA)
        devPtrs.resize(size);
#endif
    }

    size_t         iteration;                    // Current iteration
    size_t         n, side, count;               // Number of particles
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

    std::vector<KeyType> codes; // Particle Morton codes

    // For Sedov
    std::vector<T> mue, mui, temp, cv;

    T ttot, etot, ecin, eint, egrav;
    T minDt;

    std::vector<std::vector<T>*> data{&x,   &y,          &z,   &x_m1,  &y_m1, &z_m1, &vx,       &vy,       &vz,
                                      &ro,  &u,          &p,   &h,     &m,    &c,    &grad_P_x, &grad_P_y, &grad_P_z,
                                      &du,  &du_m1,      &dt,  &dt_m1, &c11,  &c12,  &c13,      &c22,      &c23,
                                      &c33, &maxvsignal, &mue, &mui,   &temp, &cv};

    const std::array<double, lt::size> wh  = lt::createWharmonicLookupTable<double, lt::size>();
    const std::array<double, lt::size> whd = lt::createWharmonicDerivativeLookupTable<double, lt::size>();

#if defined(USE_CUDA)
    sph::cuda::DeviceParticlesData<T, ParticlesData> devPtrs;

    ParticlesData()
        : devPtrs(*this){};
#endif

#ifdef USE_MPI
    MPI_Comm comm;
    int      pnamelen = 0;
    char     pname[MPI_MAX_PROCESSOR_NAME];
#endif

    int rank  = 0;
    int nrank = 1;

    // TODO: unify this with computePosition/Acceleration:
    // from SPH we have acceleration = -grad_P, so computePosition adds a factor of -1 to the pressure gradients
    // instead, the pressure gradients should be renamed to acceleration and computeMomentumAndEnergy should directly
    // set this to -grad_P, such that we don't need to add the gravitational acceleration with a factor of -1 on top
    constexpr static T g = -1.0; // for Evrard Collapse Gravity.
    // constexpr static T g = 6.6726e-8; // the REAL value of g. g is 1.0 for Evrard mainly

    constexpr static T sincIndex     = 6.0;
    constexpr static T Kcour         = 0.2;
    constexpr static T maxDtIncrease = 1.1;
    const static T     K;
};

template<typename T, typename I>
const T ParticlesData<T, I>::K = sphexa::compute_3d_k(sincIndex);

} // namespace sphexa
