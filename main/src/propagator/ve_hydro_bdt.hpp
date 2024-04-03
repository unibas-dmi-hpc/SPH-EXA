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
 * @brief A Propagator class for modern SPH with generalized volume elements
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 * @author Jose A. Escartin <ja.escartin@gmail.com>
 */

#pragma once

#include <variant>

#include "cstone/fields/field_get.hpp"
#include "sph/particles_data.hpp"
#include "sph/sph.hpp"

#include "ipropagator.hpp"
#include "gravity_wrapper.hpp"

namespace sphexa
{

using namespace sph;
using util::FieldList;

template<bool avClean, class DomainType, class DataType>
class HydroVeBdtProp : public Propagator<DomainType, DataType>
{
protected:
    using Base = Propagator<DomainType, DataType>;
    using Base::pmReader;
    using Base::timer;

    using T             = typename DataType::RealType;
    using KeyType       = typename DataType::KeyType;
    using Tmass         = typename DataType::HydroData::Tmass;
    using MultipoleType = ryoanji::CartesianQuadrupole<Tmass>;

    using Acc       = typename DataType::AcceleratorType;
    using MHolder_t = typename cstone::AccelSwitchType<Acc, MultipoleHolderCpu, MultipoleHolderGpu>::template type<
        MultipoleType, DomainType, typename DataType::HydroData>;
    template<class VType>
    using AccVector = typename cstone::AccelSwitchType<Acc, std::vector, thrust::device_vector>::template type<VType>;

    MHolder_t mHolder_;

    //! @brief spatial groups
    GroupData<Acc>        groups_;
    AccVector<float>      groupDt_;
    AccVector<LocalIndex> groupIndices_;

    //! brief timestep information rungs
    Timestep timestep_, prevTimestep_;
    //! @brief groups for each rung
    std::array<GroupData<Acc>, Timestep::maxNumRungs> rungs_;
    GroupData<Acc>                                    forceGroup_;
    GroupView                                         forceGroupView_;

    //! @brief no dependent fields can be temporarily reused as scratch space for halo exchanges
    AccVector<LocalIndex> haloRecvScratch;

    /*! @brief the list of conserved particles fields with values preserved between iterations
     *
     * x, y, z, h and m are automatically considered conserved and must not be specified in this list
     */
    using ConservedFields = FieldList<"temp", "vx", "vy", "vz", "x_m1", "y_m1", "z_m1", "du_m1", "alpha", "rung">;

    //! @brief list of dependent fields, these may be used as scratch space during domain sync
    using DependentFields_ = FieldList<"ax", "ay", "az", "prho", "c", "du", "c11", "c12", "c13", "c22", "c23", "c33",
                                       "xm", "kx", "nc", "divv", "gradh">;

    //! @brief velocity gradient fields will only be allocated when avClean is true
    using GradVFields = FieldList<"dV11", "dV12", "dV13", "dV22", "dV23", "dV33">;

    //! @brief what will be allocated based AV cleaning choice
    using DependentFields =
        std::conditional_t<avClean, decltype(DependentFields_{} + GradVFields{}), decltype(DependentFields_{})>;

    //! @brief Return rung of current block time-step
    static int activeRung(int substep, int numRungs)
    {
        if (substep == 0 || substep >= (1 << (numRungs - 1))) { return 0; }
        else { return cstone::butterfly(substep); }
    }

public:
    HydroVeBdtProp(std::ostream& output, size_t rank, const InitSettings& settings)
        : Base(output, rank)
    {
        if (avClean && rank == 0) { std::cout << "AV cleaning is activated" << std::endl; }
        try
        {
            timestep_.minDt     = settings.at("minDt");
            prevTimestep_.minDt = settings.at("minDt");
        }
        catch (const std::out_of_range&)
        {
            std::cout << "Init settings miss the following parameter: minDt" << std::endl;
            throw;
        }
    }

    std::vector<std::string> conservedFields() const override
    {
        std::vector<std::string> ret{"x", "y", "z", "h", "m"};
        for_each_tuple([&ret](auto f) { ret.push_back(f.value); }, make_tuple(ConservedFields{}));
        return ret;
    }

    void activateFields(DataType& simData) override
    {
        auto& d = simData.hydro;
        //! @brief Fields accessed in domain sync (x,y,z,h,m,keys) are not part of extensible lists.
        d.setConserved("x", "y", "z", "h", "m");
        d.setDependent("keys");
        std::apply([&d](auto... f) { d.setConserved(f.value...); }, make_tuple(ConservedFields{}));
        std::apply([&d](auto... f) { d.setDependent(f.value...); }, make_tuple(DependentFields{}));

        d.devData.setConserved("x", "y", "z", "h", "m");
        d.devData.setDependent("keys");
        std::apply([&d](auto... f) { d.devData.setConserved(f.value...); }, make_tuple(ConservedFields{}));
        std::apply([&d](auto... f) { d.devData.setDependent(f.value...); }, make_tuple(DependentFields{}));
    }

    void fullSync(DomainType& domain, DataType& simData)
    {
        auto& d = simData.hydro;
        if (d.g != 0.0)
        {
            domain.syncGrav(get<"keys">(d), get<"x">(d), get<"y">(d), get<"z">(d), get<"h">(d), get<"m">(d),
                            get<ConservedFields>(d), get<DependentFields>(d));
        }
        else
        {
            domain.sync(get<"keys">(d), get<"x">(d), get<"y">(d), get<"z">(d), get<"h">(d),
                        std::tuple_cat(std::tie(get<"m">(d)), get<ConservedFields>(d)), get<DependentFields>(d));
        }
        d.treeView = domain.octreeProperties();

        computeGroups(domain.startIndex(), domain.endIndex(), d, domain.box(), groups_);
        forceGroupView_ = groups_.view();

        reallocate(groups_.numGroups, groupDt_, groupIndices_);
        fill(groupDt_, 0, groupDt_.size(), std::numeric_limits<float>::max());
    }

    void partialSync(DomainType& domain, DataType& simData)
    {
        auto& d = simData.hydro;
        domain.exchangeHalos(get<"x", "y", "z", "h">(d), get<"keys">(d), haloRecvScratch);
        if (d.g != 0.0)
        {
            // TODO: update expansion centers
        }

        if constexpr (cstone::HaveGpu<Acc>{})
        {
            int highestRung = cstone::butterfly(timestep_.substep);
            extractGroupGpu(groups_.view(), rawPtr(groupIndices_), timestep_.rungRanges[0],
                            timestep_.rungRanges[highestRung], forceGroup_);
            forceGroupView_ = forceGroup_.view();
        }
    }

    void sync(DomainType& domain, DataType& simData) override
    {
        if (activeRung(timestep_.substep, timestep_.numRungs) == 0) { fullSync(domain, simData); }
        else { partialSync(domain, simData); }
    }

    bool isSynced() override { return activeRung(timestep_.substep, timestep_.numRungs) == 0; }

    void computeForces(DomainType& domain, DataType& simData) override
    {
        timer.start();
        pmReader.start();
        sync(domain, simData);
        timer.step("domain::sync");

        GroupView activeGroup    = forceGroupView_;
        bool      isNewHierarchy = activeRung(timestep_.substep, timestep_.numRungs) == 0;

        auto& d = simData.hydro;
        d.resize(domain.nParticlesWithHalos());
        resizeNeighbors(d, domain.nParticles() * d.ngmax);
        size_t first = domain.startIndex();
        size_t last  = domain.endIndex();

        transferToHost(d, first, first + 1, {"m"});
        fill(get<"m">(d), 0, first, d.m[first]);
        fill(get<"m">(d), last, domain.nParticlesWithHalos(), d.m[first]);

        findNeighborsSfc(first, last, d, domain.box());
        timer.step("FindNeighbors");
        pmReader.step();

        computeXMass(activeGroup, d, domain.box());
        timer.step("XMass");
        domain.exchangeHalos(std::tie(get<"xm">(d)), get<"keys">(d), haloRecvScratch);
        timer.step("mpi::synchronizeHalos");

        computeVeDefGradh(activeGroup, d, domain.box());
        timer.step("Normalization & Gradh");

        computeEOS(first, last, d);
        timer.step("EquationOfState");

        domain.exchangeHalos(get<"vx", "vy", "vz", "prho", "c", "kx">(d), get<"keys">(d), haloRecvScratch);
        timer.step("mpi::synchronizeHalos");

        computeIadDivvCurlv(activeGroup, d, domain.box());
        if (isNewHierarchy) { groupDivvTimestep(activeGroup, rawPtr(groupDt_), d); }
        timer.step("IadVelocityDivCurl");

        domain.exchangeHalos(get<"c11", "c12", "c13", "c22", "c23", "c33", "divv">(d), get<"keys">(d), haloRecvScratch);
        timer.step("mpi::synchronizeHalos");

        computeAVswitches(activeGroup, d, domain.box());
        timer.step("AVswitches");

        if (avClean)
        {
            domain.exchangeHalos(get<"dV11", "dV12", "dV22", "dV23", "dV33", "alpha">(d), get<"keys">(d),
                                 haloRecvScratch);
        }
        else { domain.exchangeHalos(std::tie(get<"alpha">(d)), get<"keys">(d), haloRecvScratch); }
        timer.step("mpi::synchronizeHalos");

        float* groupDtUse = isNewHierarchy ? rawPtr(groupDt_) : nullptr;
        computeMomentumEnergy<avClean>(activeGroup, groupDtUse, d, domain.box());
        timer.step("MomentumAndEnergy");
        pmReader.step();

        if (d.g != 0.0)
        {
            mHolder_.upsweep(d, domain);
            timer.step("Upsweep");
            pmReader.step();
            mHolder_.traverse(d, domain);
            timer.step("Gravity");
            pmReader.step();
        }

        if (isNewHierarchy) { groupAccTimestep(activeGroup, rawPtr(groupDt_), d); }
    }

    void computeBlockTimesteps(DataType& simData)
    {
        auto& d        = simData.hydro;
        int   highRung = activeRung(timestep_.substep, timestep_.numRungs);

        if (highRung == 0)
        {
            prevTimestep_  = timestep_;
            float maxIncDt = timestep_.minDt * std::pow(d.maxDtIncrease, (1 << (timestep_.numRungs - 1)));
            timestep_ = computeGroupTimestep(groups_.view(), rawPtr(groupDt_), rawPtr(groupIndices_), get<"keys">(d));
            timestep_.minDt = std::min({timestep_.minDt, maxIncDt});

            for (int r = 0; r < timestep_.numRungs; ++r)
            {
                if constexpr (cstone::HaveGpu<Acc>{})
                {
                    extractGroupGpu(groups_.view(), rawPtr(groupIndices_), timestep_.rungRanges[r],
                                    timestep_.rungRanges[r + 1], rungs_[r]);
                }
            }
        }

        if (Base::rank_ == 0 && highRung == 0)
        {
            auto ts = timestep_;

            util::array<LocalIndex, 4> numRungs = {ts.rungRanges[1], ts.rungRanges[2] - ts.rungRanges[1],
                                                   ts.rungRanges[3] - ts.rungRanges[2],
                                                   ts.rungRanges[4] - ts.rungRanges[3]};
            // clang-format off
            std::cout << "# New block-TS " << ts.numRungs << " rungs, "
                      << "R0: " << numRungs[0] << " (" << (100. * numRungs[0] / groups_.numGroups) << "%) "
                      << "R1: " << numRungs[1] << " (" << (100. * numRungs[1] / groups_.numGroups) << "%) "
                      << "R2: " << numRungs[2] << " (" << (100. * numRungs[2] / groups_.numGroups) << "%) "
                      << "R3: " << numRungs[3] << " (" << (100. * numRungs[3] / groups_.numGroups) << "%) "
                      << "All: " << groups_.numGroups << " (100%)" << std::endl;
            // clang-format on
        }
        if (Base::rank_ == 0 && highRung > 0)
        {
            LocalIndex numActiveGroups = 0;
            for (int i = 0; i < highRung; ++i)
            {
                numActiveGroups += rungs_[i].numGroups;
            }
            std::cout << "# Substep " << timestep_.substep << "/" << (1 << (timestep_.numRungs - 1)) << ", "
                      << numActiveGroups << " active groups" << std::endl;
        }

        d.ttot += timestep_.minDt;
        d.minDt_m1 = d.minDt;
        d.minDt    = timestep_.minDt;
    }

    void integrate(DomainType& domain, DataType& simData) override
    {
        auto&  d     = simData.hydro;
        size_t first = domain.startIndex();
        size_t last  = domain.endIndex();

        computeBlockTimesteps(simData);
        timer.step("Timestep");

        auto driftBack       = [](int subStep, int rung) { return subStep % (1 << rung); };
        int  lowestDriftRung = cstone::butterfly(timestep_.substep + 1);
        bool isLastSubstep   = activeRung(timestep_.substep + 1, timestep_.numRungs) == 0;
        auto substepBox      = isLastSubstep ? domain.box() : cstone::Box<T>(0, 1, cstone::BoundaryType::open);
        for (int i = 0; i < timestep_.numRungs; ++i)
        {
            int  bk      = driftBack(timestep_.substep, i);
            bool useRung = timestep_.substep == bk;
            bool advance = i < lowestDriftRung;

            float          dt      = timestep_.minDt;
            float          dt_back = dt * bk;
            float          dt_m1   = useRung ? prevTimestep_.minDt : dt * (1 << i);
            const uint8_t* rung    = useRung ? rawPtr(get<"rung">(d)) : nullptr;

            if (advance)
            {
                if (bk) { driftPositions(rungs_[i].view(), d, 0, dt_back, dt_m1, rung); }
                computePositions(rungs_[i].view(), d, substepBox, dt * (1 << i), dt_m1, rung);
            }
            else { driftPositions(rungs_[i].view(), d, dt_back + dt, dt_back, dt_m1, rung); }
        }

        updateSmoothingLength(groups_.view(), d);

        if (isLastSubstep) // if next step starts new hierarchy
        {
            for (int r = 0; r < timestep_.numRungs; ++r)
            {
                if constexpr (cstone::HaveGpu<Acc>{}) { storeRungGpu(rungs_[r].view(), r, rawPtr(get<"rung">(d))); }
            }
        }

        timestep_.substep++;
        timer.step("UpdateQuantities");
    }

    void saveFields(IFileWriter* writer, size_t first, size_t last, DataType& simData,
                    const cstone::Box<T>& box) override
    {
        auto& d             = simData.hydro;
        auto  fieldPointers = d.data();
        auto  indicesDone   = d.outputFieldIndices;
        auto  namesDone     = d.outputFieldNames;

        auto output = [&]()
        {
            for (int i = int(indicesDone.size()) - 1; i >= 0; --i)
            {
                int fidx = indicesDone[i];
                if (d.isAllocated(fidx))
                {
                    int column = std::find(d.outputFieldIndices.begin(), d.outputFieldIndices.end(), fidx) -
                                 d.outputFieldIndices.begin();
                    transferToHost(d, first, last, {d.fieldNames[fidx]});
                    std::visit([writer, c = column, key = namesDone[i]](auto field)
                               { writer->writeField(key, field->data(), c); }, fieldPointers[fidx]);
                    indicesDone.erase(indicesDone.begin() + i);
                    namesDone.erase(namesDone.begin() + i);
                }
            }
        };

        // first output pass: write everything allocated at the end of the step
        output();

        if (!indicesDone.empty() && Base::rank_ == 0)
        {
            std::cout << "WARNING: the following fields are not in use and therefore not output: ";
            for (int fidx = 0; fidx < indicesDone.size() - 1; ++fidx)
            {
                std::cout << d.fieldNames[fidx] << ",";
            }
            std::cout << d.fieldNames[indicesDone.back()] << std::endl;
        }
    }

    void saveExtra(IFileWriter* writer, DataType& /*simData*/) override
    {
        if constexpr (cstone::HaveGpu<Acc>{})
        {
            auto               numGroups = groupDt_.size();
            std::vector<float> h_groupDt(numGroups);
            memcpyD2H(rawPtr(groupDt_), numGroups, h_groupDt.data());

            writer->addStep(0, numGroups, "group_dt.h5");
            writer->writeField("dt", h_groupDt.data(), 0);
            writer->closeStep();
        }
    }
};

} // namespace sphexa
