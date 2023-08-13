#pragma once

#include <iostream>
#include "sph/particles_data.hpp"
#include "propagator/ipropagator.hpp"
#include "cstone/domain/domain.hpp"

#ifdef SPH_EXA_USE_CATALYST2
#include "catalyst_adaptor.h"
#endif

#ifdef SPH_EXA_USE_ASCENT
#include "ascent_adaptor.h"
#endif

namespace viz
{

void init_catalyst([[maybe_unused]] int argc, [[maybe_unused]] char** argv)
{
#ifdef SPH_EXA_USE_CATALYST2
    CatalystAdaptor::Initialize(argc, argv);
    std::cout << "CatalystInitialize\n";
#endif
}

template<class DataType>
void init_ascent([[maybe_unused]] DataType& d, [[maybe_unused]] long startIndex)
{
#ifdef SPH_EXA_USE_ASCENT
    AscentAdaptor::Initialize(d, startIndex);
    std::cout << "AscentInitialize\n";
#endif
}

template<class DataType, class DomainType, class ParticleDataType>
void execute([[maybe_unused]] DataType&                                                          d,
             [[maybe_unused]] std::unique_ptr<sphexa::Propagator<DomainType, ParticleDataType>>& p,
             [[maybe_unused]] long startIndex, [[maybe_unused]] long endIndex, [[maybe_unused]] size_t rank)
{
#ifdef SPH_EXA_USE_CATALYST2
    CatalystAdaptor::Execute(d, startIndex, endIndex);
#endif
#ifdef SPH_EXA_USE_ASCENT
    AscentAdaptor::Execute(d, p, startIndex, endIndex, rank);
#endif
}

void finalize()
{
#ifdef SPH_EXA_USE_CATALYST2
    CatalystAdaptor::Finalize();
#endif
#ifdef SPH_EXA_USE_ASCENT
    AscentAdaptor::Finalize();
#endif
}

} // namespace viz
