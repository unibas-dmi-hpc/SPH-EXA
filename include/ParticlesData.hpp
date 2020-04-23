#pragma once

#include <cstdio>
#include <vector>
#include "BBox.hpp"
#include "sph/kernels.hpp"

namespace sphexa
{
template <typename T>
struct ParticlesData
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
    std::vector<T> vol;                          // Volume
    // todo: it's redundant to carry around volume, mass and density... how to resolve this???
    //       which 2 of the 3 do we keep? Or is comms and memory overhead negligible?
    std::vector<T> xmass;                        // to store X_a (VE estimator function). We need it because we only update at end of run...
    std::vector<T> sumkx;                        // kernel weighted sum of VE estimators (sumkx in sphynx)
    std::vector<T> ballmass;                     // this is needed to do newton-raphson for h and density
    std::vector<T> sumwh;                        // this is needed to calculate the derivative of the ballmass[i]/h[i]**3 - ro[i] = 0 with newton-raphson
    std::vector<T> u;                            // Internal Energy
    std::vector<T> p, p_0;                       // Pressure
    std::vector<T> h;                            // Smoothing Length
    std::vector<T> m;                            // Mass
    std::vector<T> c;                            // Speed of sound
    std::vector<T> grad_P_x, grad_P_y, grad_P_z; // gradient of the pressure
    std::vector<T> du, du_m1;                    // variation of the energy without AV contribution
    std::vector<T> du_av, du_av_m1;              // artificial viscosity contribution of the variation of the energy
    std::vector<T> dt, dt_m1;
    std::vector<T> c11, c12, c13, c22, c23, c33; // IAD components
    std::vector<T> maxvsignal;
    // todo: the nn ones should be reconciled with the neighborscount in the task struct.
    //       problem is that like this here, we can dump it easily for debug, but they're actually uints, not doubles!
    //       challenge for the dump: I'm not sure how the particle index from the task clist maps to the index in the
    //       particle data arrays (basically the i = clist[pi] mapping). If we want to dump data from the task struct,
    //       we need to tell the writeparticledata function which index to use... (whether to apply the mapping or not)
    std::vector<T> nn;                           // number of neighbors
    std::vector<T> nn_actual;                    // actual number of neighbors (not capped by ngmax)
    std::vector<T> gradh;                        // gradh terms (omega)
    std::vector<T> id;                           // particle identifier (index is reordere). this should be globally unique
    std::vector<T> volnorm;                      // to check volume normalization (integration of volume should be 1)
    std::vector<T> avgdeltar_x;                  // to check the \sum_b{V_b(r_b-r_a)*W_{ab} = 0 condition (cabezon2017 Fig5 right)
    std::vector<T> avgdeltar_y;                  // to check the \sum_b{V_b(r_b-r_a)*W_{ab} = 0 condition (cabezon2017 Fig5 right)
    std::vector<T> avgdeltar_z;                  // to check the \sum_b{V_b(r_b-r_a)*W_{ab} = 0 condition (cabezon2017 Fig5 right)

    T ttot, etot, ecin, eint;
    T minDt;

    // for windblob
    T masscloudinic;                             // to get initial mass of cloud
    T masscloud;
    T rocloud;
    T uambient;
    T tkh;

    BBox<T> bbox;

    std::vector<std::vector<T> *> data{&x,        &y,        &z,   &x_m1,       &y_m1,     &z_m1,   &vx,    &vy,    &vz,
                                       &ro,       &ro_0,     &u,   &p,          &p_0,      &h,      &m,     &c,     &grad_P_x,
                                       &grad_P_y, &grad_P_z, &du,  &du_m1,      &dt,       &dt_m1,  &c11,   &c12,   &c13,
                                       &c22,      &c23,      &c33, &maxvsignal, &vol,      &xmass,  &sumkx, &sumwh, &ballmass,
                                       &nn,       &gradh,    &id,  &du_av,      &du_av_m1, &volnorm, &avgdeltar_x,
                                       &avgdeltar_y, &avgdeltar_z, &nn_actual};
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

    // general VE
    constexpr static T veExp = 0.7;

    // whether to use old AV implmentation (gringold monoghan 1983 or newer one
    bool oldAV = false;


#ifndef NDEBUG
    bool writeErrorOnNegU = false;
#endif
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
