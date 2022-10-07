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

#include <filesystem>
#include <sstream>
#include <variant>

#include "cstone/fields/particles_get.hpp"
#include "sph/sph.hpp"
#include "sph/hydro_turb/turbulence_data.hpp"

#include "io/mpi_file_utils.hpp"

#include "ipropagator.hpp"
#include "ve_hydro.hpp"
#include "gravity_wrapper.hpp"

namespace sphexa
{

using namespace sph;
using cstone::FieldStates;

//! @brief VE hydro propagator that adds turbulence stirring to the acceleration prior to position update
template<class DomainType, class DataType>
class TurbVeProp final : public HydroVeProp<DomainType, DataType>
{
    using Base = HydroVeProp<DomainType, DataType>;
    using Base::ng0_;
    using Base::ngmax_;
    using Base::rank_;
    using Base::timer;

    using RealType = typename DataType::RealType;

    sph::TurbulenceData<RealType, typename DataType::AcceleratorType> turbulenceData;

public:
    TurbVeProp(size_t ngmax, size_t ng0, std::ostream& output, size_t rank)
        : Base(ngmax, ng0, output, rank)
        , turbulenceData(TurbulenceConstants(), rank == 0)
    {
    }

    void restoreState(const std::string& path, MPI_Comm comm) override
    {
        // The file does not exist, we're starting from scratch. Nothing to do.
        if (!std::filesystem::exists(path)) { return; }

        H5PartFile* h5_file = nullptr;
        h5_file             = fileutils::openH5Part(path, H5PART_READ, comm);
        size_t numSteps     = H5PartGetNumSteps(h5_file);

        if (numSteps == 0) { throw std::runtime_error("Cannot initialize phases from empty file\n"); }

        size_t h5_step = numSteps - 1;
        H5PartSetStep(h5_file, h5_step);

        size_t iteration;
        H5PartReadStepAttrib(h5_file, "step", &iteration);
        auto fileAttributes = fileutils::fileAttributeNames(h5_file);

        {
            // restore turbulence mode phases
            std::string attrName = "phases_" + std::to_string(iteration);
            size_t      attrIndex =
                std::find(fileAttributes.begin(), fileAttributes.end(), attrName) - fileAttributes.begin();
            if (attrIndex == fileAttributes.size()) { throw std::runtime_error("No data found at " + attrName); }

            h5part_int64_t typeId, attrSize;
            char           dummy[256];
            H5PartGetFileAttribInfo(h5_file, attrIndex, dummy, 256, &typeId, &attrSize);

            if (attrSize != turbulenceData.phases.size())
            {
                throw std::runtime_error("Stored number of phases does not match initialized number of phases\n");
            }

            h5part_int64_t errors = H5PART_SUCCESS;
            errors |= H5PartReadFileAttrib(h5_file, attrName.c_str(), turbulenceData.phases.data());

            if (errors != H5PART_SUCCESS) { throw std::runtime_error("Could not read turbulence phases\n"); }
        }
        {
            // restore random number engine state
            std::string attrName = "rngEngineState_" + std::to_string(iteration);
            size_t      attrIndex =
                std::find(fileAttributes.begin(), fileAttributes.end(), attrName) - fileAttributes.begin();
            if (attrIndex == fileAttributes.size()) { throw std::runtime_error("No data found at " + attrName); }

            h5part_int64_t typeId, attrSize;
            char           dummy[256];
            H5PartGetFileAttribInfo(h5_file, attrIndex, dummy, 256, &typeId, &attrSize);

            char engineState[attrSize];

            h5part_int64_t errors = H5PART_SUCCESS;
            errors |= H5PartReadFileAttrib(h5_file, attrName.c_str(), engineState);

            if (errors != H5PART_SUCCESS) { throw std::runtime_error("Could not read engine state\n"); }

            std::stringstream s;
            s << engineState;
            s >> turbulenceData.gen;
        }

        if (rank_ == 0) { std::cout << "Restored phases and RNG state from SPH iteration " << iteration << std::endl; }

        H5PartCloseFile(h5_file);
    }

    void step(DomainType& domain, DataType& simData) override
    {
        Base::computeForces(domain, simData);

        auto&  d     = simData.hydro;
        size_t first = domain.startIndex();
        size_t last  = domain.endIndex();

        computeTimestep(d);
        timer.step("Timestep");
        driveTurbulence(first, last, d, turbulenceData);
        timer.step("Turbulence Stirring");

        computePositions(first, last, d, domain.box());
        timer.step("UpdateQuantities");
        updateSmoothingLength(first, last, d, ng0_);
        timer.step("UpdateSmoothingLength");

        timer.stop();
    }

    //! @brief save turbulence mode phases to file
    void dump(size_t iteration, const std::string& path) override
    {
        // turbulence phases are identical on all ranks, only rank 0 needs to write data
        if (rank_ > 0) { return; }

        H5PartFile* h5_file = nullptr;
        h5_file             = H5PartOpenFile(path.c_str(), H5PART_APPEND);
        {
            std::string attributeName = "phases_" + std::to_string(iteration);
            const auto& phases        = turbulenceData.phases;
            fileutils::sphexaWriteFileAttrib(h5_file, attributeName.c_str(), phases.data(), phases.size());
        }
        {
            std::stringstream s;
            s << turbulenceData.gen;
            std::string engineState = s.str();

            std::string attributeName = "rngEngineState_" + std::to_string(iteration);
            fileutils::sphexaWriteFileAttrib(h5_file, attributeName.c_str(), engineState.data(), engineState.size());
        }

        H5PartCloseFile(h5_file);
    }
};

} // namespace sphexa
