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

#include "Timer.hpp"
#include "Task.hpp"
#include "ParticlesData.hpp"

#include "cstone/domain/domain.hpp"

#include "sph/timestep.hpp"

using namespace cstone;
using namespace sphexa;
using namespace sphexa::sph;

class Propagator
{

private:

    const size_t       nTasks;                                  //
    const size_t       ngmax;                                   //
    const size_t       ng0;                                     //

public:

    TaskList           taskList;                                //
    MasterProcessTimer timer;                                   //

    Propagator(
        const size_t  nTasks,                                   //
        const size_t  ngmax,                                    //
        const size_t  ng0,                                      //
        const size_t  nParticles,                               //
        std::ostream& output,                                   //
        const size_t  rank)                                     //
        : nTasks(nTasks)
        , ngmax(ngmax)
        , ng0(ng0)
        , taskList(0, nParticles, nTasks, ngmax, ng0)
        , timer(output, rank)
    {}

    size_t neighbors(){
        return neighborsSum(taskList.tasks);
    }

    template <class DomainType, class ParticleDataType>
    void hydroStep(DomainType& domain, ParticleDataType& d)
    {
        // Advance simulation by one step
        using T = ParticleDataType::RealType;

        timer.start();

        domain.sync(d.codes, d.x, d.y, d.z, d.h, d.m, d.mui, d.u, d.vx, d.vy, d.vz, d.x_m1, d.y_m1, d.z_m1, d.du_m1,d.dt_m1);
        timer.step("domain::sync");

        d.resize(domain.nParticlesWithHalos());

        std::fill(begin(d.m),                     begin(d.m) + domain.startIndex(),          d.m[domain.startIndex()]);
        std::fill(begin(d.m) + domain.endIndex(), begin(d.m) + domain.nParticlesWithHalos(), d.m[domain.startIndex()]);

        taskList.update(domain.startIndex(), domain.endIndex());
        timer.step("updateTasks");

        findNeighborsSfc(taskList.tasks, d.x, d.y, d.z, d.h, d.codes, domain.box());
        timer.step("FindNeighbors");

        computeDensity<T>(taskList.tasks, d, domain.box());
        timer.step("Density");

        computeEquationOfStateEvrard<T>(taskList.tasks, d);
        timer.step("EquationOfState");

        domain.exchangeHalos(d.vx, d.vy, d.vz, d.ro, d.p, d.c);
        timer.step("mpi::synchronizeHalos");

        computeIAD<T>(taskList.tasks, d, domain.box());
        timer.step("IAD");

        domain.exchangeHalos(d.c11, d.c12, d.c13, d.c22, d.c23, d.c33);
        timer.step("mpi::synchronizeHalos");

        computeMomentumAndEnergyIAD<T>(taskList.tasks, d, domain.box());
        timer.step("MomentumEnergyIAD");

        computeTimestep<T, TimestepPress2ndOrder<T, ParticleDataType>>(taskList.tasks, d);
        timer.step("Timestep");

        computePositions<T, computeAcceleration<T, ParticleDataType>>(taskList.tasks, d, domain.box());
        timer.step("UpdateQuantities");

        computeTotalEnergy<T>(taskList.tasks, d);
        d.etot += d.egrav;
        timer.step("EnergyConservation");

        updateSmoothingLength<T>(taskList.tasks, d);
        timer.step("UpdateSmoothingLength");

        timer.stop();
    }


    template <class DomainType, class ParticleDataType>
    void hydroStepGravity(DomainType& domain, ParticleDataType& d)
    {
        // Advance simulation by one step
        using T = ParticleDataType::RealType;

        timer.start();

        domain.sync(d.codes, d.x, d.y, d.z, d.h, d.m, d.mui, d.u, d.vx, d.vy, d.vz, d.x_m1, d.y_m1, d.z_m1, d.du_m1,d.dt_m1);
        timer.step("domain::sync");

        d.resize(domain.nParticlesWithHalos());

        std::fill(begin(d.m),                     begin(d.m) + domain.startIndex(),          d.m[domain.startIndex()]);
        std::fill(begin(d.m) + domain.endIndex(), begin(d.m) + domain.nParticlesWithHalos(), d.m[domain.startIndex()]);

        taskList.update(domain.startIndex(), domain.endIndex());
        timer.step("updateTasks");

        findNeighborsSfc(taskList.tasks, d.x, d.y, d.z, d.h, d.codes, domain.box());
        timer.step("FindNeighbors");

        computeDensity<T>(taskList.tasks, d, domain.box());
        timer.step("Density");

        computeEquationOfStateEvrard<T>(taskList.tasks, d);
        timer.step("EquationOfState");

        domain.exchangeHalos(d.vx, d.vy, d.vz, d.ro, d.p, d.c);
        timer.step("mpi::synchronizeHalos");

        computeIAD<T>(taskList.tasks, d, domain.box());
        timer.step("IAD");

        domain.exchangeHalos(d.c11, d.c12, d.c13, d.c22, d.c23, d.c33);
        timer.step("mpi::synchronizeHalos");

        computeMomentumAndEnergyIAD<T>(taskList.tasks, d, domain.box());
        timer.step("MomentumEnergyIAD");

        d.egrav = domain.addGravityAcceleration(d.x, d.y, d.z, d.h, d.m, d.g, d.grad_P_x, d.grad_P_y, d.grad_P_z);
        // temporary sign fix, see note in ParticlesData
        d.egrav = (d.g > 0.0) ? d.egrav : -d.egrav;
        timer.step("Gravity");

        computeTimestep<T, TimestepPress2ndOrder<T, ParticleDataType>>(taskList.tasks, d);
        timer.step("Timestep");

        computePositions<T, computeAcceleration<T, ParticleDataType>>(taskList.tasks, d, domain.box());
        timer.step("UpdateQuantities");

        computeTotalEnergy<T>(taskList.tasks, d);
        d.etot += d.egrav;
        timer.step("EnergyConservation");

        updateSmoothingLength<T>(taskList.tasks, d);
        timer.step("UpdateSmoothingLength");

        timer.stop();
    }
};
