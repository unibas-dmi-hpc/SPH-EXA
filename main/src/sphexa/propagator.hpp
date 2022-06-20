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
#include <variant>

#include "cstone/domain/domain.hpp"
#include "sph/sph.hpp"
#include "sph/traits.hpp"
#include "util/timer.hpp"
#include "gravity_wrapper.hpp"
#include "physics/turbulence/driver_turbulence.hpp"

namespace sphexa
{

using namespace sphexa::sph;

template<class DomainType, class ParticleDataType>
class Propagator
{
public:
    Propagator(size_t ngmax, size_t ng0, std::ostream& output, size_t rank, bool doGrav)
        : timer(output, rank)
        , out(output)
        , rank_(rank)
        , ngmax_(ngmax)
        , ng0_(ng0)
        , doGravity_(doGrav)
    {
    }

    virtual void sync(DomainType& domain, ParticleDataType& d) = 0;

    virtual void step(DomainType& domain, ParticleDataType& d) = 0;

    virtual void prepareOutput(ParticleDataType& d, size_t startIndex, size_t endIndex){};

    virtual ~Propagator() = default;

protected:
    MasterProcessTimer timer;
    std::ostream&      out;

    size_t rank_;
    //! maximum number of neighbors per particle
    size_t ngmax_;
    //! target number of neighbors per particle
    size_t ng0_;

    bool doGravity_;

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
    using Base::doGravity_;
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
    HydroProp(size_t ngmax, size_t ng0, std::ostream& output, size_t rank, bool doGravity)
        : Base(ngmax, ng0, output, rank, doGravity)
    {
    }

    void sync(DomainType& domain, ParticleDataType& d) override
    {
        domain.sync(d.codes, d.x, d.y, d.z, d.h, d.m, d.u, d.vx, d.vy, d.vz, d.x_m1, d.y_m1, d.z_m1, d.du_m1);
    }

    void step(DomainType& domain, ParticleDataType& d) override
    {
        timer.start();
        sync(domain, d);
        timer.step("domain::sync");

        resize(d, domain.nParticlesWithHalos());
        resizeNeighbors(d, domain.nParticles() * ngmax_);
        size_t first = domain.startIndex();
        size_t last  = domain.endIndex();

        std::fill(begin(d.m), begin(d.m) + first, d.m[first]);
        std::fill(begin(d.m) + last, end(d.m), d.m[first]);

        findNeighborsSfc<T, KeyType>(
            first, last, ngmax_, d.x, d.y, d.z, d.h, d.codes, d.neighbors, d.neighborsCount, domain.box());
        timer.step("FindNeighbors");
        computeDensity(first, last, ngmax_, d, domain.box());
        timer.step("Density");
        computeEquationOfState3L(first, last, d);
        timer.step("EquationOfState");
        domain.exchangeHalos(d.vx, d.vy, d.vz, d.rho, d.p, d.c);
        timer.step("mpi::synchronizeHalos");
        computeIAD(first, last, ngmax_, d, domain.box());
        timer.step("IAD");
        domain.exchangeHalos(d.c11, d.c12, d.c13, d.c22, d.c23, d.c33);
        timer.step("mpi::synchronizeHalos");
        computeMomentumAndEnergy(first, last, ngmax_, d, domain.box());
        timer.step("MomentumEnergyIAD");

/*#ifdef USE_CUDA
            size_t sizeWithHalos = d.x.size();
            size_t size_np_T     = sizeWithHalos * sizeof(decltype(d.grad_P_x[0]));
            CHECK_CUDA_ERR(cudaMemcpy(d.grad_P_x.data(), d.devPtrs.d_grad_P_x, size_np_T, cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERR(cudaMemcpy(d.grad_P_y.data(), d.devPtrs.d_grad_P_y, size_np_T, cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERR(cudaMemcpy(d.grad_P_z.data(), d.devPtrs.d_grad_P_z, size_np_T, cudaMemcpyDeviceToHost));
#endif
*/
        //llamar driver  size_np_T=npart grad_P_x   (mirar compute_density)   //box.xmax-box.xmin

        computeTimestep(first, last, d);
        timer.step("Timestep");
        computePositions(first, last, d, domain.box());
        timer.step("UpdateQuantities");
        computeTotalEnergy(first, last, d);
        timer.step("EnergyConservation");
        updateSmoothingLength(first, last, d, ng0_);
        timer.step("UpdateSmoothingLength");

        timer.stop();
        this->printIterationTimings(domain, d);
    }
};

template<class DomainType, class ParticleDataType>
class HydroVeProp final : public Propagator<DomainType, ParticleDataType>
{
    using Base = Propagator<DomainType, ParticleDataType>;
    using Base::doGravity_;
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
    HydroVeProp(size_t ngmax, size_t ng0, std::ostream& output, size_t rank, bool doGravity)
        : Base(ngmax, ng0, output, rank, doGravity)
    {
    }

    void sync(DomainType& domain, ParticleDataType& d) override
    {
        if (doGravity_)
        {
            domain.syncGrav(
                d.codes, d.x, d.y, d.z, d.h, d.m, d.u, d.vx, d.vy, d.vz, d.x_m1, d.y_m1, d.z_m1, d.du_m1, d.alpha);
        }
        else
        {
            domain.sync(
                d.codes, d.x, d.y, d.z, d.h, d.m, d.u, d.vx, d.vy, d.vz, d.x_m1, d.y_m1, d.z_m1, d.du_m1, d.alpha);
        }
    }

    void step(DomainType& domain, ParticleDataType& d) override
    {
        timer.start();
        sync(domain, d);
        timer.step("domain::sync");

        resize(d, domain.nParticlesWithHalos());
        resizeNeighbors(d, domain.nParticles() * ngmax_);
        size_t first = domain.startIndex();
        size_t last  = domain.endIndex();

        std::fill(begin(d.m), begin(d.m) + first, d.m[first]);
        std::fill(begin(d.m) + last, end(d.m), d.m[first]);

        findNeighborsSfc<T, KeyType>(
            first, last, ngmax_, d.x, d.y, d.z, d.h, d.codes, d.neighbors, d.neighborsCount, domain.box());
        timer.step("FindNeighbors");

        computeXMass(first, last, ngmax_, d, domain.box());
        timer.step("XMass");
        domain.exchangeHalos(d.xm);
        timer.step("mpi::synchronizeHalos");
        computeDensityVE(first, last, ngmax_, d, domain.box());
        timer.step("Density & Gradh");
        computeEquationOfState(first, last, d);
        timer.step("EquationOfState");
        domain.exchangeHalos(d.vx, d.vy, d.vz, d.p, d.c, d.kx, d.gradh);
        timer.step("mpi::synchronizeHalos");
        computeIadDivvCurlv(first, last, ngmax_, d, domain.box());
        timer.step("IadVelocityDivCurl");
        domain.exchangeHalos(d.c11, d.c12, d.c13, d.c22, d.c23, d.c33, d.divv, d.curlv);
        timer.step("mpi::synchronizeHalos");
        computeAVswitches(first, last, ngmax_, d, domain.box());
        timer.step("AVswitches");
        domain.exchangeHalos(d.alpha);
        timer.step("mpi::synchronizeHalos");
        computeGradPVE(first, last, ngmax_, d, domain.box());
        timer.step("MomentumAndEnergy");

        d.egrav = 0.0;
        if (doGravity_)
        {
            mHolder_.upsweep(d, domain);
            timer.step("Upsweep");
            mHolder_.traverse(d, domain);
            // temporary sign fix, see note in ParticlesData
            d.egrav = (d.g > 0.0) ? d.egrav : -d.egrav;
            timer.step("Gravity");
        }

        computeTimestep(first, last, d);
        timer.step("Timestep");
        computePositions(first, last, d, domain.box());
        timer.step("UpdateQuantities");
        computeTotalEnergy(first, last, d);
        timer.step("EnergyConservation");
        updateSmoothingLength(first, last, d, ng0_);
        timer.step("UpdateSmoothingLength");

        timer.stop();
        this->printIterationTimings(domain, d);
    }

    void prepareOutput(ParticleDataType& d, size_t startIndex, size_t endIndex) override
    {
        auto fieldPointers = getOutputArrays(d);

        for (size_t outIdx = 0; outIdx < fieldPointers.size(); ++outIdx)
        {
            if (d.outputFieldNames[outIdx] == "rho")
            {
                auto computeRho =
                    [startIndex, endIndex, kx = d.kx.data(), m = d.m.data(), xm = d.xm.data()](auto* varField)
                {
#pragma omp parallel for schedule(static)
                    for (size_t i = startIndex; i < endIndex; ++i)
                    {
                        varField[i] = kx[i] * m[i] / xm[i];
                    }
                };

                std::visit(computeRho, fieldPointers[outIdx]);
            }
        }
    }
};

template<class DomainType, class ParticleDataType>
class TurbProp final : public Propagator<DomainType, ParticleDataType>
{
    using Base = Propagator<DomainType, ParticleDataType>;
    using Base::doGravity_;
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
    TurbProp(size_t ngmax, size_t ng0, std::ostream& output, size_t rank, bool doGravity)
        : Base(ngmax, ng0, output, rank, doGravity)
    {
    }

    void sync(DomainType& domain, ParticleDataType& d) override
    {
        if (doGravity_)
        {
            domain.syncGrav(
                d.codes, d.x, d.y, d.z, d.h, d.m, d.u, d.vx, d.vy, d.vz, d.x_m1, d.y_m1, d.z_m1, d.du_m1, d.alpha);
        }
        else
        {
            domain.sync(
                d.codes, d.x, d.y, d.z, d.h, d.m, d.u, d.vx, d.vy, d.vz, d.x_m1, d.y_m1, d.z_m1, d.du_m1, d.alpha);
        }
    }

    void step(DomainType& domain, ParticleDataType& d) override
    {
        timer.start();
        sync(domain, d);
        timer.step("domain::sync");

        resize(d, domain.nParticlesWithHalos());
        resizeNeighbors(d, domain.nParticles() * ngmax_);
        size_t first = domain.startIndex();
        size_t last  = domain.endIndex();

        std::fill(begin(d.m), begin(d.m) + first, d.m[first]);
        std::fill(begin(d.m) + last, end(d.m), d.m[first]);

        findNeighborsSfc<T, KeyType>(
            first, last, ngmax_, d.x, d.y, d.z, d.h, d.codes, d.neighbors, d.neighborsCount, domain.box());
        timer.step("FindNeighbors");

        computeXMass(first, last, ngmax_, d, domain.box());
        timer.step("XMass");
        domain.exchangeHalos(d.xm);
        timer.step("mpi::synchronizeHalos");
        computeDensityVE(first, last, ngmax_, d, domain.box());
        timer.step("Density & Gradh");
        computeEquationOfState(first, last, d);
        timer.step("EquationOfState");
        domain.exchangeHalos(d.vx, d.vy, d.vz, d.p, d.c, d.kx, d.gradh);
        timer.step("mpi::synchronizeHalos");
        computeIadDivvCurlv(first, last, ngmax_, d, domain.box());
        timer.step("IadVelocityDivCurl");
        domain.exchangeHalos(d.c11, d.c12, d.c13, d.c22, d.c23, d.c33, d.divv, d.curlv);
        timer.step("mpi::synchronizeHalos");
        computeAVswitches(first, last, ngmax_, d, domain.box());
        timer.step("AVswitches");
        domain.exchangeHalos(d.alpha);
        timer.step("mpi::synchronizeHalos");
        computeGradPVE(first, last, ngmax_, d, domain.box());
        timer.step("MomentumAndEnergy");

        d.egrav = 0.0;
        if (doGravity_)
        {
            mHolder_.upsweep(d, domain);
            timer.step("Upsweep");
            mHolder_.traverse(d, domain);
            // temporary sign fix, see note in ParticlesData
            d.egrav = (d.g > 0.0) ? d.egrav : -d.egrav;
            timer.step("Gravity");
        }

        computeTimestep(first, last, d);
        timer.step("Timestep");
        driver_turbulence(first,last,d);
        timer.step("Turbulence Stirring");
        computePositions(first, last, d, domain.box());
        timer.step("UpdateQuantities");
        computeTotalEnergy(first, last, d);
        timer.step("EnergyConservation");
        updateSmoothingLength(first, last, d, ng0_);
        timer.step("UpdateSmoothingLength");

        timer.stop();
        this->printIterationTimings(domain, d);
    }

    void prepareOutput(ParticleDataType& d, size_t startIndex, size_t endIndex) override
    {
        auto fieldPointers = getOutputArrays(d);

        for (size_t outIdx = 0; outIdx < fieldPointers.size(); ++outIdx)
        {
            if (d.outputFieldNames[outIdx] == "rho")
            {
                auto computeRho =
                    [startIndex, endIndex, kx = d.kx.data(), m = d.m.data(), xm = d.xm.data()](auto* varField)
                {
#pragma omp parallel for schedule(static)
                    for (size_t i = startIndex; i < endIndex; ++i)
                    {
                        varField[i] = kx[i] * m[i] / xm[i];
                    }
                };

                std::visit(computeRho, fieldPointers[outIdx]);
            }
        }
    }
};

template<class DomainType, class ParticleDataType>
std::unique_ptr<Propagator<DomainType, ParticleDataType>>
propagatorFactory(bool ve, size_t ngmax, size_t ng0, std::ostream& output, size_t rank, bool doGravity)
{
    if (ve) { return std::make_unique<HydroVeProp<DomainType, ParticleDataType>>(ngmax, ng0, output, rank, doGravity); }
    else { return std::make_unique<HydroProp<DomainType, ParticleDataType>>(ngmax, ng0, output, rank, doGravity); }
}

} // namespace sphexa
