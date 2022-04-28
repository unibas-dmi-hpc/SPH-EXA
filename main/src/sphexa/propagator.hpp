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
 * @brief A Propagator class to manage the loop for each the timestep decoupled of the tests
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 * @author Jose A. Escartin <ja.escartin@gmail.com>
 */

#pragma once

#include <iostream>

#include "cstone/domain/domain.hpp"
#include "sph/sph.hpp"
#include "sph/traits.hpp"
#include "util/timer.hpp"

#include "gravity_wrapper.hpp"

namespace sphexa
{

using namespace sphexa::sph;

template<class DomainType, class ParticleDataType>
class Propagator
{
public:
    Propagator(size_t ngmax, size_t ng0, std::ostream& output, size_t rank)
        : timer(output, rank)
        , out(output)
        , rank_(rank)
        , ngmax_(ngmax)
        , ng0_(ng0)
    {
    }

    virtual void step(DomainType& domain, ParticleDataType& d) = 0;

    virtual ~Propagator() = default;

protected:
    MasterProcessTimer timer;
    std::ostream&      out;

    size_t rank_;
    //! maximum number of neighbors per particle
    size_t ngmax_;
    //! average number of neighbors per particle
    size_t ng0_;

    void printIterationTimings(const DomainType& domain, const ParticleDataType& d)
    {
        size_t totalNeighbors = neighborsSum(domain.startIndex(), domain.endIndex(), d.neighborsCount);

        if (rank_ == 0)
        {
            printCheck(d.ttot,
                       d.minDt,
                       d.etot,
                       d.eint,
                       d.ecin,
                       d.egrav,
                       domain.box(),
                       d.numParticlesGlobal,
                       domain.nParticles(),
                       domain.globalTree().numLeafNodes(),
                       domain.nParticlesWithHalos() - domain.nParticles(),
                       totalNeighbors);

            std::cout << "### Check ### Focus Tree Nodes: " << domain.focusTree().octree().numLeafNodes() << std::endl;
            printTotalIterationTime(d.iteration, timer.duration());
        }
    }

    void printTotalIterationTime(size_t iteration, float duration)
    {
        out << "=== Total time for iteration(" << iteration << ") " << duration << "s" << std::endl << std::endl;
    }

    template<class Box>
    void printCheck(double totalTime, double minTimeStep, double totalEnergy, double internalEnergy,
                    double kineticEnergy, double gravitationalEnergy, const Box& box, size_t totalParticleCount,
                    size_t particleCount, size_t nodeCount, size_t haloCount, size_t totalNeighbors)
    {
        out << "### Check ### Global Tree Nodes: " << nodeCount << ", Particles: " << particleCount
            << ", Halos: " << haloCount << std::endl;
        out << "### Check ### Computational domain: " << box.xmin() << " " << box.xmax() << " " << box.ymin() << " "
            << box.ymax() << " " << box.zmin() << " " << box.zmax() << std::endl;
        out << "### Check ### Total Neighbors: " << totalNeighbors
            << ", Avg neighbor count per particle: " << totalNeighbors / totalParticleCount << std::endl;
        out << "### Check ### Total time: " << totalTime << ", current time-step: " << minTimeStep << std::endl;
        out << "### Check ### Total energy: " << totalEnergy << ", (internal: " << internalEnergy
            << ", cinetic: " << kineticEnergy;
        out << ", gravitational: " << gravitationalEnergy;
        out << ")" << std::endl;
    }
};

template<class DomainType, class ParticleDataType>
class HydroProp final : public Propagator<DomainType, ParticleDataType>
{
    using Base = Propagator<DomainType, ParticleDataType>;
    using Base::ng0_;
    using Base::ngmax_;
    using Base::timer;

    using T             = typename ParticleDataType::RealType;
    using KeyType       = typename ParticleDataType::KeyType;
    using MultipoleType = ryoanji::CartesianQuadrupole<float>;

    using Acc = typename ParticleDataType::AcceleratorType;
    using MHolder_t =
        typename detail::AccelSwitchType<Acc, MultipoleHolderCpu, MultipoleHolderGpu>::template type<MultipoleType,
                                                                                                     KeyType, T, T, T>;

    MHolder_t mHolder_;

public:
    HydroProp(size_t ngmax, size_t ng0, std::ostream& output, size_t rank)
        : Base(ngmax, ng0, output, rank)
    {
    }

    void step(DomainType& domain, ParticleDataType& d) override
    {
        timer.start();

#ifdef USE_CUDA
        size_t sizeWithHalos    = d.x.size();
        size_t size_np_T        = sizeWithHalos * sizeof(T);
        size_t size_np_CodeType = sizeWithHalos * sizeof(KeyType);
#endif

        bool doGrav = (d.g != 0.0);
        if (doGrav)
        {
            domain.syncGrav(d.codes, d.x, d.y, d.z, d.h, d.m, d.u, d.vx, d.vy, d.vz, d.x_m1, d.y_m1, d.z_m1, d.du_m1);
        }
        else
        {
            domain.sync(d.codes, d.x, d.y, d.z, d.h, d.m, d.u, d.vx, d.vy, d.vz, d.x_m1, d.y_m1, d.z_m1, d.du_m1);
        }
        timer.step("domain::sync: codes, x,y,z, h, m, u, vx,vy,vz, x_m1,y_m1,z_m1, du_m1");

        resize(d, domain.nParticlesWithHalos());
        resizeNeighbors(d, domain.nParticles() * ngmax_);

        size_t first = domain.startIndex();
        size_t last  = domain.endIndex();

        std::fill(begin(d.m), begin(d.m) + first, d.m[first]);
        std::fill(begin(d.m) + last, end(d.m), d.m[first]);

        findNeighborsSfc<T, KeyType>(
            first, last, ngmax_, d.x, d.y, d.z, d.h, d.codes, d.neighbors, d.neighborsCount, domain.box());
        timer.step("FindNeighbors");
#ifdef USE_CUDA
        CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_codes, d.codes.data(), size_np_CodeType, cudaMemcpyHostToDevice));
        timer.step("  * GPU CudaCopyIn Sync HostToDevice: codes");
#endif

#ifdef USE_CUDA
        CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_x, d.x.data(), size_np_T, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_y, d.y.data(), size_np_T, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_z, d.z.data(), size_np_T, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_h, d.h.data(), size_np_T, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_m, d.m.data(), size_np_T, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_vx, d.vx.data(), size_np_T, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_vy, d.vy.data(), size_np_T, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_vz, d.vz.data(), size_np_T, cudaMemcpyHostToDevice));
        timer.step("  * GPU CudaCopyIn Sync HostToDevice: x,y,z, h, m, vx,vy,vz");
#endif
        computeRhoZero(first, last, ngmax_, d, domain.box());
#ifdef USE_CUDA
        CHECK_CUDA_ERR(cudaMemcpy(d.rho0.data(), d.devPtrs.d_rho0, size_np_T, cudaMemcpyDeviceToHost));
#endif
        timer.step("RhoZero");
        domain.exchangeHalos(d.rho0);
        timer.step("  + mpi::synchronizeHalos: rho0");
#ifdef USE_CUDA
        CHECK_CUDA_ERR(cudaMemcpy(d.rho0.data(), d.devPtrs.d_rho0, size_np_T, cudaMemcpyDeviceToHost));
        timer.step("  * GPU CudaCopyBack Sync DeviceToHost: rho0");
#endif

        computeDensity(first, last, ngmax_, d, domain.box());
#ifdef USE_CUDA
        CHECK_CUDA_ERR(cudaMemcpy(d.rho.data(), d.devPtrs.d_rho, size_np_T, cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERR(cudaMemcpy(d.kx.data(), d.devPtrs.d_kx, size_np_T, cudaMemcpyDeviceToHost));
#endif
        timer.step("Density");
        domain.exchangeHalos(d.rho, d.kx);
        timer.step("  + mpi::synchronizeHalos: rho, kx");
#ifdef USE_CUDA
        CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_rho, d.rho.data(), size_np_T, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_kx, d.kx.data(), size_np_T, cudaMemcpyHostToDevice));
        timer.step("  * GPU CudaCopyIn Sync HostToDevice: rho, kx");
#endif

        computeEquationOfState(first, last, d);
        timer.step("EquationOfState");
        domain.exchangeHalos(d.p, d.c);
        timer.step("  + mpi::synchronizeHalos: p, c");
#ifdef USE_CUDA
        CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_p, d.p.data(), size_np_T, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_c, d.c.data(), size_np_T, cudaMemcpyHostToDevice));
        timer.step("  * GPU CudaCopyIn Sync HostToDevice: p, c");
#endif

        computeIAD(first, last, ngmax_, d, domain.box());
#ifdef USE_CUDA
        CHECK_CUDA_ERR(cudaMemcpy(d.c11.data(), d.devPtrs.d_c11, size_np_T, cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERR(cudaMemcpy(d.c12.data(), d.devPtrs.d_c12, size_np_T, cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERR(cudaMemcpy(d.c13.data(), d.devPtrs.d_c13, size_np_T, cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERR(cudaMemcpy(d.c22.data(), d.devPtrs.d_c22, size_np_T, cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERR(cudaMemcpy(d.c23.data(), d.devPtrs.d_c23, size_np_T, cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERR(cudaMemcpy(d.c33.data(), d.devPtrs.d_c33, size_np_T, cudaMemcpyDeviceToHost));
#endif
        timer.step("IAD");
        domain.exchangeHalos(d.c11, d.c12, d.c13, d.c22, d.c23, d.c33);
        timer.step("  + mpi::synchronizeHalos: c11, c12, c13, c22, c23, c33");
#ifdef USE_CUDA
        CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_c11, d.c11.data(), size_np_T, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_c12, d.c12.data(), size_np_T, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_c13, d.c13.data(), size_np_T, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_c22, d.c22.data(), size_np_T, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_c23, d.c23.data(), size_np_T, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_c33, d.c33.data(), size_np_T, cudaMemcpyHostToDevice));
        timer.step("  * GPU CudaCopyIn Sync HostToDevice: c11, c12, c13, c22, c23, c33");
#endif

        computeDivvCurlv(first, last, ngmax_, d, domain.box());
#ifdef USE_CUDA
        CHECK_CUDA_ERR(cudaMemcpy(d.divv.data(), d.devPtrs.d_divv, size_np_T, cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERR(cudaMemcpy(d.curlv.data(), d.devPtrs.d_curlv, size_np_T, cudaMemcpyDeviceToHost));
#endif
        timer.step("VelocityDivCurl");
        domain.exchangeHalos(d.divv, d.curlv);
        timer.step("  + mpi::synchronizeHalos: divv, curlv");
#ifdef USE_CUDA
        CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_divv, d.divv.data(), size_np_T, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_curlv, d.curlv.data(), size_np_T, cudaMemcpyHostToDevice));
        timer.step("  * GPU CudaCopyIn Sync HostToDevice: divv, curlv");
#endif

        computeAVswitches(first, last, ngmax_, d, domain.box());
#ifdef USE_CUDA
        CHECK_CUDA_ERR(cudaMemcpy(d.alpha.data(), d.devPtrs.d_alpha, size_np_T, cudaMemcpyDeviceToHost));
#endif
        timer.step("AVswitches");
        domain.exchangeHalos(d.alpha);
        timer.step("  + mpi::synchronizeHalos: alpha");
#ifdef USE_CUDA
        CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_alpha, d.alpha.data(), size_np_T, cudaMemcpyHostToDevice));
        timer.step("  * GPU CudaCopyIn Sync HostToDevice: alpha");
#endif

        computeMomentumEnergy(first, last, ngmax_, d, domain.box());
#ifdef USE_CUDA
        CHECK_CUDA_ERR(cudaMemcpy(d.du.data(), d.devPtrs.d_du, size_np_T, cudaMemcpyDeviceToHost));

        if (d.g == 0)
        {
            CHECK_CUDA_ERR(cudaMemcpy(d.grad_P_x.data(), d.devPtrs.d_grad_P_x, size_np_T, cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERR(cudaMemcpy(d.grad_P_y.data(), d.devPtrs.d_grad_P_y, size_np_T, cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERR(cudaMemcpy(d.grad_P_z.data(), d.devPtrs.d_grad_P_z, size_np_T, cudaMemcpyDeviceToHost));
        }
#endif
        timer.step("MomentumEnergy");

        d.egrav = 0.0;
        if (doGrav)
        {
            mHolder_.upsweep(d, domain);
            timer.step("Upsweep");

            mHolder_.traverse(d, domain);

            // temporary sign fix, see note in ParticlesData
            d.egrav = (d.g > 0.0) ? d.egrav : -d.egrav;

#ifdef USE_CUDA
            CHECK_CUDA_ERR(cudaMemcpy(d.grad_P_x.data(), d.devPtrs.d_grad_P_x, size_np_T, cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERR(cudaMemcpy(d.grad_P_y.data(), d.devPtrs.d_grad_P_y, size_np_T, cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERR(cudaMemcpy(d.grad_P_z.data(), d.devPtrs.d_grad_P_z, size_np_T, cudaMemcpyDeviceToHost));
#endif
            timer.step("Gravity");
        }

        computeTimestep(first, last, d);
        timer.step("Timestep");

        computeUpdateQuantities(first, last, d, domain.box());
        timer.step("UpdateQuantities");

        computeEnergyConservation(first, last, d);
        timer.step("EnergyConservation");

        computeSmoothingLength(first, last, d, ng0_);
        timer.step("SmoothingLength");

        timer.stop();
        this->printIterationTimings(domain, d);
    }
};

template<class DomainType, class ParticleDataType>
std::unique_ptr<Propagator<DomainType, ParticleDataType>> propagatorFactory(size_t ngmax, size_t ng0,
                                                                            std::ostream& output, size_t rank)
{
    return std::make_unique<HydroProp<DomainType, ParticleDataType>>(ngmax, ng0, output, rank);
}

} // namespace sphexa
