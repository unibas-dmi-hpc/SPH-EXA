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

#include "timer.hpp"
#include "task.hpp"
#include "particles_data.hpp"

#include "cstone/domain/domain.hpp"
#include "ryoanji/cpu/treewalk.hpp"
#include "ryoanji/cpu/upsweep.hpp"

#include "sph/timestep.hpp"

using namespace cstone;
using namespace sphexa;
using namespace sphexa::sph;

class Propagator
{
public:
    Propagator(const size_t nTasks, const size_t ngmax, const size_t ng0, std::ostream& output, const size_t rank)
        : taskList(nTasks, ngmax, ng0)
        , timer(output, rank)
        , output_(output)
    {
    }

    //! @brief Advance simulation by one step with hydro-dynamical forces
    template<class DomainType, class ParticleDataType>
    void hydroStep(DomainType& domain, ParticleDataType& d)
    {
        using T       = typename ParticleDataType::RealType;
        using KeyType = typename ParticleDataType::KeyType;

        timer.start();

        domain.sync(d.codes, d.x, d.y, d.z, d.h, d.m, d.u, d.vx, d.vy, d.vz, d.x_m1, d.y_m1, d.z_m1, d.du_m1, d.dt_m1);
        timer.step("domain::sync");

        resize(d, domain.nParticlesWithHalos());

        std::fill(begin(d.m), begin(d.m) + domain.startIndex(), d.m[domain.startIndex()]);
        std::fill(begin(d.m) + domain.endIndex(), begin(d.m) + domain.nParticlesWithHalos(), d.m[domain.startIndex()]);

        taskList.update(domain.startIndex(), domain.endIndex());
        timer.step("updateTasks");

        findNeighborsSfc<T, KeyType>(taskList.tasks, d.x, d.y, d.z, d.h, d.codes, d.neighborsCount, domain.box());
        timer.step("FindNeighbors");

        computeDensity<T>(taskList.tasks, d, domain.box());
        timer.step("Density");

        computeEquationOfState(domain.startIndex(), domain.endIndex(), d);
        timer.step("EquationOfState");

        domain.exchangeHalos(d.vx, d.vy, d.vz, d.rho, d.p, d.c);
        timer.step("mpi::synchronizeHalos");

        computeIAD<T>(taskList.tasks, d, domain.box());
        timer.step("IAD");

        domain.exchangeHalos(d.c11, d.c12, d.c13, d.c22, d.c23, d.c33);
        timer.step("mpi::synchronizeHalos");

        computeMomentumAndEnergyIAD<T>(taskList.tasks, d, domain.box());
        timer.step("MomentumEnergyIAD");

        computeTimestep(domain.startIndex(), domain.endIndex(), d);
        timer.step("Timestep");

        computePositions(domain.startIndex(), domain.endIndex(), d, domain.box());
        timer.step("UpdateQuantities");

        d.egrav = 0;
        computeTotalEnergy<T>(domain.startIndex(), domain.endIndex(), d);
        timer.step("EnergyConservation");

        updateSmoothingLength<T>(taskList.tasks, d);
        timer.step("UpdateSmoothingLength");

        timer.stop();

        printIterationTimings(domain, d);
    }

    //! @brief Advance simulation by one step with hydro-dynamical and self-gravity forces
    template<class DomainType, class ParticleDataType>
    void hydroStepGravity(DomainType& domain, ParticleDataType& d)
    {
        using T             = typename ParticleDataType::RealType;
        using KeyType       = typename ParticleDataType::KeyType;
        using MultipoleType = ryoanji::CartesianQuadrupole<T>;

        timer.start();

        domain.syncGrav(
            d.codes, d.x, d.y, d.z, d.h, d.m, d.u, d.vx, d.vy, d.vz, d.x_m1, d.y_m1, d.z_m1, d.du_m1, d.dt_m1);
        timer.step("domain::sync");

        resize(d, domain.nParticlesWithHalos());

        std::fill(begin(d.m), begin(d.m) + domain.startIndex(), d.m[domain.startIndex()]);
        std::fill(begin(d.m) + domain.endIndex(), begin(d.m) + domain.nParticlesWithHalos(), d.m[domain.startIndex()]);

        taskList.update(domain.startIndex(), domain.endIndex());
        timer.step("updateTasks");

        findNeighborsSfc<T, KeyType>(taskList.tasks, d.x, d.y, d.z, d.h, d.codes, d.neighborsCount, domain.box());
        timer.step("FindNeighbors");

        computeDensity<T>(taskList.tasks, d, domain.box());
        timer.step("Density");

        computeEquationOfState(domain.startIndex(), domain.endIndex(), d);
        timer.step("EquationOfState");

        domain.exchangeHalos(d.vx, d.vy, d.vz, d.rho, d.p, d.c);
        timer.step("mpi::synchronizeHalos");

        computeIAD<T>(taskList.tasks, d, domain.box());
        timer.step("IAD");

        domain.exchangeHalos(d.c11, d.c12, d.c13, d.c22, d.c23, d.c33);
        timer.step("mpi::synchronizeHalos");

        computeMomentumAndEnergyIAD<T>(taskList.tasks, d, domain.box());
        timer.step("MomentumEnergyIAD");

        const Octree<KeyType>&               octree  = domain.focusTree();
        gsl::span<const SourceCenterType<T>> centers = domain.expansionCenters();

        std::vector<MultipoleType> multipoles(octree.numTreeNodes());
        ryoanji::computeLeafMultipoles(
            octree, domain.layout(), d.x.data(), d.y.data(), d.z.data(), d.m.data(), centers.data(), multipoles.data());

        ryoanji::CombineMultipole<MultipoleType> combineMultipole(centers.data());
        //! first upsweep with local data
        upsweep(octree, multipoles.data(), combineMultipole);
        domain.template exchangeFocusGlobal<MultipoleType>(multipoles, combineMultipole);
        //! second upsweep with leaf data from peer ranks in place
        upsweep(octree, multipoles.data(), combineMultipole);

        d.egrav = ryoanji::computeGravity(octree,
                                          centers.data(),
                                          multipoles.data(),
                                          domain.layout().data(),
                                          domain.startCell(),
                                          domain.endCell(),
                                          d.x.data(),
                                          d.y.data(),
                                          d.z.data(),
                                          d.h.data(),
                                          d.m.data(),
                                          d.g,
                                          d.grad_P_x.data(),
                                          d.grad_P_y.data(),
                                          d.grad_P_z.data());

        // temporary sign fix, see note in ParticlesData
        d.egrav = (d.g > 0.0) ? d.egrav : -d.egrav;
        timer.step("Gravity");

        computeTimestep(domain.startIndex(), domain.endIndex(), d);
        timer.step("Timestep");

        computePositions(domain.startIndex(), domain.endIndex(), d, domain.box());
        timer.step("UpdateQuantities");

        computeTotalEnergy<T>(domain.startIndex(), domain.endIndex(), d);
        timer.step("EnergyConservation");

        updateSmoothingLength<T>(taskList.tasks, d);
        timer.step("UpdateSmoothingLength");

        timer.stop();

        printIterationTimings(domain, d);
    }

private:
    TaskList           taskList;
    MasterProcessTimer timer;
    std::ostream&      output_;

    template<class DomainType, class ParticleDataType>
    void printIterationTimings(const DomainType& domain, const ParticleDataType& d)
    {
        size_t totalNeighbors = neighborsSum(domain.startIndex(), domain.endIndex(), d.neighborsCount);

        if (d.rank == 0)
        {
            Printer::printCheck(d.ttot,
                                d.minDt,
                                d.etot,
                                d.eint,
                                d.ecin,
                                d.egrav,
                                domain.box(),
                                d.n,
                                domain.nParticles(),
                                nNodes(domain.tree()),
                                domain.nParticlesWithHalos() - domain.nParticles(),
                                totalNeighbors,
                                output_);

            std::cout << "### Check ### Focus Tree Nodes: " << domain.focusTree().numLeafNodes() << std::endl;
            Printer::printTotalIterationTime(d.iteration, timer.duration(), output_);
        }
    }
};
