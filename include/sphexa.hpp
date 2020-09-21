#pragma once

#ifdef USE_MPI
#include "mpi.h"
#include "DistributedDomain.hpp"
#endif

#include "debugUtils.hpp"

#include "Domain.hpp"
#include "Octree.hpp"
#include "BBox.hpp"
#include "Task.hpp"
#include "ArgParser.hpp"
#include "Timer.hpp"
#include "FileUtils.hpp"
#include "Printer.hpp"
#include "utils.hpp"

#if defined(USE_CUDA)
#error "The code was refactored to support General Volume Elements, but the CUDA code was not addressed yet."
#include "sph/cuda/sph.cuh"
#endif

#if defined(USE_ACC) || defined(USE_OMP_TARGET)
#error "The code was refactored to support General Volume Elements, but openACC and OMP Target have not been addressed yet."
#endif

#include "sph/findNeighbors.hpp"
#include "sph/density.hpp"
#include "sph/newtonRaphson.hpp"
#include "sph/gradhTerms.hpp"
#include "sph/IAD.hpp"
#include "sph/momentumAndEnergyIAD.hpp"
#include "sph/kernels.hpp"
#include "sph/equationOfState.hpp"
#include "sph/timestep.hpp"
#include "sph/positions.hpp"
#include "sph/totalEnergy.hpp"
#include "sph/updateSmoothingLength.hpp"
#include "sph/updateVEEstimator.hpp"
#include "sph/gravityTreeWalk.hpp"
#include "sph/gravityTreeWalkForRemoteParticles.hpp"
