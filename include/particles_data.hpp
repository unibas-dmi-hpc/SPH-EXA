#pragma once

#include <array>
#include <cstdio>
#include <vector>

#include "sph/kernels.hpp"
#include "sph/tables.hpp"

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

    size_t iteration;      // Current iteration
    size_t n, side, count; // Number of particles

    T ttot, etot, ecin, eint, egrav;
    T minDt;

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
    std::vector<T> mue, mui, temp, cv;

    std::vector<KeyType> codes; // Particle space-filling-curve keys

    /*! @brief
     * Name of each field as string for use e.g in HDF5 output. Order has to correspond to what's returned by data().
     */
    inline static constexpr std::array fieldNames{
        "x",   "y",   "z",   "x_m1", "y_m1",     "z_m1",     "vx",         "vy",  "vz",    "ro",   "u",
        "p",   "h",   "m",   "c",    "grad_P_x", "grad_P_y", "grad_P_z",   "du",  "du_m1", "dt",   "dt_m1",
        "c11", "c12", "c13", "c22",  "c23",      "c33",      "maxvsignal", "mue", "mui",   "temp", "cv"};

    /*! @brief return a vector of pointers to field vectors
     *
     * We implement this by returning an rvalue to prevent having to store pointers and avoid
     * non-trivial copy/move constructors.
     */
    auto data()
    {
        std::array<std::vector<T>*, 33> ret{
            &x,   &y,   &z,   &x_m1, &y_m1,     &z_m1,     &vx,         &vy,  &vz,    &ro,   &u,
            &p,   &h,   &m,   &c,    &grad_P_x, &grad_P_y, &grad_P_z,   &du,  &du_m1, &dt,   &dt_m1,
            &c11, &c12, &c13, &c22,  &c23,      &c33,      &maxvsignal, &mue, &mui,   &temp, &cv};

        static_assert(ret.size() == fieldNames.size());

        return ret;
    }

    std::vector<std::string> outputFields;

    const std::array<double, lt::size> wh  = lt::createWharmonicLookupTable<double, lt::size>();
    const std::array<double, lt::size> whd = lt::createWharmonicDerivativeLookupTable<double, lt::size>();

#if defined(USE_CUDA)
    sph::cuda::DeviceParticlesData<T, ParticlesData> devPtrs;

    ParticlesData()
        : devPtrs(*this){};
#endif

#ifdef USE_MPI
    MPI_Comm comm;
#endif
    int rank  = 0;
    int nrank = 1;

    // TODO: unify this with computePosition/Acceleration:
    // from SPH we have acceleration = -grad_P, so computePosition adds a factor of -1 to the pressure gradients
    // instead, the pressure gradients should be renamed to acceleration and computeMomentumAndEnergy should directly
    // set this to -grad_P, such that we don't need to add the gravitational acceleration with a factor of -1 on top
    T g = -1.0; // for Evrard Collapse Gravity.
    // constexpr static T g = 6.6726e-8; // the REAL value of g. g is 1.0 for Evrard mainly

    constexpr static T sincIndex     = 6.0;
    constexpr static T Kcour         = 0.2;
    constexpr static T maxDtIncrease = 1.1;
    const static T     K;
};

template<typename T, typename I>
const T ParticlesData<T, I>::K = sphexa::compute_3d_k(sincIndex);


//! @brief resizes all particles fields of @p d listed in data() to the specified size
template<class Dataset>
void resize(Dataset& d, size_t size)
{
    double growthRate = 1.05;
    auto data_ = d.data();

    for (size_t i = 0; i < data_.size(); ++i)
    {
        reallocate(*data_[i], size, growthRate);
    }
    reallocate(d.codes, size, growthRate);

#if defined(USE_CUDA)
    d.devPtrs.resize(size);
#endif
}

//! @brief construct a vector of pointers to particle fields for file output
template<class Dataset>
auto getOutputArrays(Dataset& dataset, const std::vector<std::string>& fields)
{
    using T = typename Dataset::RealType;

    auto fieldPointers = dataset.data();
    auto fieldNames = Dataset::fieldNames;

    std::vector<const T*> outputFields;

    for (const auto& field : fields)
    {
        auto it = std::find(fieldNames.begin(), fieldNames.end(), field);

        if (it == fieldNames.end()) { throw std::runtime_error("Cannot output field " + field + "\n"); }

        size_t fieldIndex = it - fieldNames.begin();
        outputFields.push_back(fieldPointers[fieldIndex]->data());
    }

    return outputFields;
}

} // namespace sphexa
