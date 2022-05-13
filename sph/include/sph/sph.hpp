#pragma once

#if defined(USE_CUDA)
#include "sph/cuda/sph.cuh"
#endif

#include "sph/find_neighbors.hpp"

#include "sph/density.hpp"
#include "sph/iad.hpp"
#include "sph/momentum_energy.hpp"
#include "sph/kernels.hpp"
#include "sph/eos.hpp"
#include "sph/timestep.hpp"
#include "sph/positions.hpp"
#include "sph/total_energy.hpp"
#include "sph/update_h.hpp"

#include "sph/av_switches.hpp"
#include "sph/density_ve.hpp"
#include "sph/iad_divv_curlv.hpp"
#include "sph/momentum_energy_ve.hpp"
#include "sph/xmass.hpp"