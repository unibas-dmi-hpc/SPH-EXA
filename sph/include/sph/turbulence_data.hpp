
#pragma once

#include "particles_data.hpp"
namespace sphexa
{

template<typename T, typename I, class AccType>
class TurbulenceData : public ParticlesData<T, I, AccType>
{
public:

    //constants_
    T stSolWeight;

    size_t stSeed;                 // seed for random number generator             //
    size_t stNModes;               // Number of computed nodes
    size_t ndim;
    T stOUvar;
    T stDecay;
    T stSolWeightNorm;

    std::vector<T> stOUPhases;
    std::vector<T> stMode;
    std::vector<T> stAmpl;
};


} // namespace sphexa
