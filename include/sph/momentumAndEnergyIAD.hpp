#pragma once

#include <vector>

#include <cmath>
#include "math.hpp"
#include "kernels.hpp"
#include "lookupTables.hpp"
#include "cuda/sph.cuh"

#if defined(USE_CUDA) || defined(USE_ACC) || defined(USE_OMP_TARGET)
#error "The code was refactored to support General Volume Elements, but accelerator code has not been addressed yet."
#endif

namespace sphexa
{
namespace sph
{

template <typename T, class Dataset>
void computeMomentumAndEnergyIADImpl(const Task &t, Dataset &d)
{
    const size_t n = t.clist.size();
    const size_t ngmax = t.ngmax;
    const int *clist = t.clist.data();
    const int *neighbors = t.neighbors.data();
    const int *neighborsCount = t.neighborsCount.data();

    const T *h = d.h.data();
    const T *m = d.m.data();
    const T *x = d.x.data();
    const T *y = d.y.data();
    const T *z = d.z.data();
    const T *vx = d.vx.data();
    const T *vy = d.vy.data();
    const T *vz = d.vz.data();
    const T *ro = d.ro.data();
    const T *c = d.c.data();
    const T *p = d.p.data();

    const T *c11 = d.c11.data();
    const T *c12 = d.c12.data();
    const T *c13 = d.c13.data();
    const T *c22 = d.c22.data();
    const T *c23 = d.c23.data();
    const T *c33 = d.c33.data();

    T *du = d.du.data();
    T *du_av = d.du_av.data();
    T *du_m1 = d.du_m1.data();
    T *du_av_m1 = d.du_av_m1.data();

    T *grad_P_x = d.grad_P_x.data();
    T *grad_P_y = d.grad_P_y.data();
    T *grad_P_z = d.grad_P_z.data();
    T *maxvsignal = d.maxvsignal.data();
    T *volnorm = d.volnorm.data();  // check kernel normalization
    T *avgdeltar_x = d.avgdeltar_x.data(); // check <delta r> magnitude
    T *avgdeltar_y = d.avgdeltar_y.data();
    T *avgdeltar_z = d.avgdeltar_z.data();

    const BBox<T> bbox = d.bbox;

    const T K = d.K;
    const T sincIndex = d.sincIndex;

    // general VE
    const T *sumkx = d.sumkx.data();
    const T *xmass = d.xmass.data();
    const T *vol = d.vol.data();

    // gradh
    const T *gradh = d.gradh.data();

#if defined(USE_OMP_TARGET)
    // Apparently Cray with -O2 has a bug when calling target regions in a loop. (and computeMomentumAndEnergyIADImpl can be called in a
    // loop). A workaround is to call some method or allocate memory to either prevent buggy optimization or other side effect. with -O1
    // there is no problem Tested with Cray 8.7.3 with NVIDIA Tesla P100 on PizDaint
    std::vector<T> imHereBecauseOfCrayCompilerO2Bug(4, 10);

    const int np = d.x.size();
    const size_t allNeighbors = n * ngmax;
// clang-format off
#pragma omp target map(to:                                                                                                          \
                        clist[:n], neighbors[:allNeighbors], neighborsCount[:n], x[0:np], y[0:np], z[0:np],                        \
                        vx[0:np], vy[0:np], vz[0:np], h[0:np], m[0:np], ro[0:np], p[0:np], c[0:np],                                 \
                        c11[0:np], c12[0:np], c13[0:np], c22[0:np], c23[0:np], c33[0:np])                                           \
                   map(from                                                                                                         \
                       :grad_P_x[:np], grad_P_y[:np], grad_P_z[:np], du[:np], maxvsignal[:np])
// clang-format on
#pragma omp teams distribute parallel for
#elif defined(USE_ACC)
    const int np = d.x.size();
    const size_t allNeighbors = n * ngmax;
// clang-format off
#pragma acc parallel loop copyin(clist[0:n], neighbors[0:allNeighbors], neighborsCount[0:n], x[0:np], y[0:np], z[0:np], vx[0:np],   \
                                 vy[0:np], vz[0:np], h[0:np], m [0:np], ro[0:np], p[0:np], c[0:np], c11[0:np], c12[0:np],            \
                                 c13[0:np], c22[0:np], c23[0:np], c33[0:np])                                                         \
                          copyout(grad_P_x[:np], grad_P_y[:np], grad_P_z[:np], du[:np], maxvsignal[:np])
// clang-format on
#else
#pragma omp parallel for schedule(guided)
#endif
    for (size_t pi = 0; pi < n; ++pi)
    {
        const int i = clist[pi];
        const int nn = neighborsCount[pi];

        const T gradh_i = gradh[i];

        T maxvsignali = 0.0;
        T momentum_x = 0.0, momentum_y = 0.0, momentum_z = 0.0, energy = 0.0, energyAV = 0.0;
        T volnorm_loc = 0.0;
        T avgdeltar_x_loc = 0.0, avgdeltar_y_loc = 0.0, avgdeltar_z_loc = 0.0;

        for (int pj = 0; pj < nn; ++pj)
        {
            const int j = neighbors[pi * ngmax + pj];

            const T gradh_j = gradh[j];

            T r_ijx = (x[i] - x[j]);
            T r_ijy = (y[i] - y[j]);
            T r_ijz = (z[i] - z[j]);

            T r_jix = (x[j] - x[i]);
            T r_jiy = (y[j] - y[i]);
            T r_jiz = (z[j] - z[i]);

            applyPBC(bbox, 2.0 * h[i], r_ijx, r_ijy, r_ijz);
            applyPBC(bbox, 2.0 * h[i], r_jix, r_jiy, r_jiz);

            const T dist = std::sqrt(r_ijx * r_ijx + r_ijy * r_ijy + r_ijz * r_ijz);

            const T v_ijx = (vx[i] - vx[j]);
            const T v_ijy = (vy[i] - vy[j]);
            const T v_ijz = (vz[i] - vz[j]);

            const T v1 = dist / h[i];
            const T v2 = dist / h[j];

            const T rv = r_ijx * v_ijx + r_ijy * v_ijy + r_ijz * v_ijz;

            const T w1 = K * math_namespace::pow(wharmonic(v1), (int)sincIndex);
            const T w2 = K * math_namespace::pow(wharmonic(v2), (int)sincIndex);

            const T W1 = w1 / (h[i] * h[i] * h[i]);
            const T W2 = w2 / (h[j] * h[j] * h[j]);

            const T kern11_i = c11[i] * r_jix;
            const T kern12_i = c12[i] * r_jiy;
            const T kern13_i = c13[i] * r_jiz;
            const T kern21_i = c12[i] * r_jix;
            const T kern22_i = c22[i] * r_jiy;
            const T kern23_i = c23[i] * r_jiz;
            const T kern31_i = c13[i] * r_jix;
            const T kern32_i = c23[i] * r_jiy;
            const T kern33_i = c33[i] * r_jiz;

            const T kern11_j = c11[j] * r_jix;
            const T kern12_j = c12[j] * r_jiy;
            const T kern13_j = c13[j] * r_jiz;
            const T kern21_j = c12[j] * r_jix;
            const T kern22_j = c22[j] * r_jiy;
            const T kern23_j = c23[j] * r_jiz;
            const T kern31_j = c13[j] * r_jix;
            const T kern32_j = c23[j] * r_jiy;
            const T kern33_j = c33[j] * r_jiz;

            const T termA1_i = (kern11_i + kern12_i + kern13_i) * W1;
            const T termA2_i = (kern21_i + kern22_i + kern23_i) * W1;
            const T termA3_i = (kern31_i + kern32_i + kern33_i) * W1;

            const T termA1_j = (kern11_j + kern12_j + kern13_j) * W2;
            const T termA2_j = (kern21_j + kern22_j + kern23_j) * W2;
            const T termA3_j = (kern31_j + kern32_j + kern33_j) * W2;

            const T pro_i = p[i] / (gradh_i * sumkx[i] * sumkx[i]);  // Pa/(omega * Kappa^2), in cabezon2017, eqs 22&23 todo: consider moving this out of the loop (only i-terms)
            const T pro_j = p[j] / (gradh_j * sumkx[j] * sumkx[j]);

            const T r_square = dist * dist;

            // For time-step calculations
            const T wij = rv / dist;  // todo: what if dist calc is wrong? See problem with PBC...
            const T vijsignal = c[i] + c[j] - 3.0 * wij;
            if (vijsignal > maxvsignali) maxvsignali = vijsignal;

            T viscosity_ij;
            if (d.oldAV){
                viscosity_ij = artificial_viscosity(ro[i], ro[j], h[i], h[j], c[i], c[j], rv, r_square);
            }
            else {
                const T alpha = 4.0 / 3.0; // sphynx parameters.f90
                viscosity_ij = rv < 0.0 ? - alpha * 0.5 * wij * vijsignal : 0.0; // cabezon2017 eq27
            }


            const T grad_Px_AV = 0.5 * (vol[i] / m[i] * m[j] * viscosity_ij * termA1_i + vol[j] * viscosity_ij * termA1_j);  // cabezon2017 eq29
            const T grad_Py_AV = 0.5 * (vol[i] / m[i] * m[j] * viscosity_ij * termA2_i + vol[j] * viscosity_ij * termA2_j);
            const T grad_Pz_AV = 0.5 * (vol[i] / m[i] * m[j] * viscosity_ij * termA3_i + vol[j] * viscosity_ij * termA3_j);

            momentum_x += xmass[i] / m[i] * xmass[j] * (pro_i * termA1_i + pro_j * termA1_j) + grad_Px_AV; // cabezon2017 eq22
            momentum_y += xmass[i] / m[i] * xmass[j] * (pro_i * termA2_i + pro_j * termA2_j) + grad_Py_AV;
            momentum_z += xmass[i] / m[i] * xmass[j] * (pro_i * termA3_i + pro_j * termA3_j) + grad_Pz_AV;

            energy += xmass[j] * pro_i * (v_ijx * termA1_i + v_ijy * termA2_i + v_ijz * termA3_i); // cabezon2017 eq 23
            energyAV += grad_Px_AV * v_ijx + grad_Py_AV * v_ijy + grad_Pz_AV * v_ijz;  // cabezon2017 eq 23

            T voljW = vol[j] * W1;
            volnorm_loc += voljW;
            avgdeltar_x_loc += voljW * r_jix;
            avgdeltar_y_loc += voljW * r_jiy;
            avgdeltar_z_loc += voljW * r_jiz;
        }
//        todo: check sphynx where the additional 0.5 is for the viscosity energy... couldn't find it after
//          brief look in momeqnmod.f90 and update.f90

        du_m1[i] = du[i];
        du_av_m1[i] = du_av[i];

        du[i] = xmass[i] / m[i] * energy; // cabezon2017 eq 32.. sphynx seems to have an extra *2.0 for the energy (line 219, momeqnmod.f90)
        du_av[i] = std::max(0.0, 0.5 * energyAV); // tried removing 0.5 because couldn't find it in sphynx. still crash at 200. The 0.5 here might serve to get the same ratio between energy and energy AV as in sphynx
                                                  // sphynx has an extra 0.5 for the energy and avist (lines 182, 183 in update.f90). These should cancel, right?
                                                  // BUT: sphynx seems to have a du_av[i] = max(0.0, energyAV) -> du_av can't be negative. I have added that

        grad_P_x[i] = momentum_x;
        grad_P_y[i] = momentum_y;
        grad_P_z[i] = momentum_z;
        maxvsignal[i] = maxvsignali;

        volnorm[i] = volnorm_loc + vol[i] * K / (h[i] * h[i] * h[i]); // self contrib and store volume normalization
        avgdeltar_x[i] = avgdeltar_x_loc;  //no self contrib because deltar = 0...
        avgdeltar_y[i] = avgdeltar_y_loc;
        avgdeltar_z[i] = avgdeltar_z_loc;
    }
}

template <typename T, class Dataset>
void computeMomentumAndEnergyIAD(const std::vector<Task> &taskList, Dataset &d)
{
#if defined(USE_CUDA)
    cuda::computeMomentumAndEnergyIAD<T>(taskList, d);
#else
    for (const auto &task : taskList)
    {
        computeMomentumAndEnergyIADImpl<T>(task, d);
    }
#endif
}

} // namespace sph
} // namespace sphexa
