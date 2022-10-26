/*
 * MIT License
 *
 * Copyright (c) 2022 CSCS, ETH Zurich, University of Basel, University of Zurich
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
 * @brief Contains the object holding all simulation data
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <mpi.h>

#include "cooling/chemistry_data.hpp"
#include "cooling/cooling.hpp"
#include "sph/particles_data.hpp"

namespace sphexa
{

//! @brief the place to store hydro, chemistry, nuclear and other simulation data
template<typename T, typename KeyType_, class AccType>
class SimulationData
{
public:
    using AcceleratorType = AccType;
    using KeyType         = KeyType_;
    using RealType        = T;

    using HydroData = ParticlesData<RealType, KeyType, AccType>;
    using ChemData  = cooling::ChemistryData<T>;


    //! @brief spacially distributed data for hydrodynamics and gravity
    HydroData hydro;

    //! @brief chemistry data for radiative cooling, e.g. for GRACKLE
    ChemData chem;
    
    //! @brief non-spacially distributed nuclear abundances
    // NuclearData nuclear;

    MPI_Comm comm;
    SimulationData(const std::string &grackle_options_file_path,
                   const double ms_sim = 1e16,
                   const double kp_sim = 46400.,
                   const int comoving_coordinates = 0,
                   const std::optional<double> t_sim = std::nullopt):
                   chem(grackle_options_file_path, ms_sim, kp_sim, comoving_coordinates, t_sim) {};
};

} // namespace sphexa
