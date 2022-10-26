//
// Created by Noah Kubli on 26.10.22.
//

#ifndef SPHEXA_EVRARD_COOLING_INIT_HPP
#define SPHEXA_EVRARD_COOLING_INIT_HPP

#include "evrard_init.hpp"
#include "cooling/cooling.hpp"

template<class Dataset>
class EvrardGlassSphereCooling : public sphexa::EvrardGlassSphere<Dataset>
{
public:
    EvrardGlassSphereCooling(std::string initBlock):
            sphexa::EvrardGlassSphere<Dataset>(initBlock)
    {}

    cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, size_t cbrtNumPart,
                                                 Dataset& simData) const override
    {
        auto box = sphexa::EvrardGlassSphere<Dataset>::init(rank, numRanks, cbrtNumPart, simData);
        cooling::initGrackleData(simData.chem, simData.hydro.x.size());
        return box;
    }
};



#endif //SPHEXA_EVRARD_COOLING_INIT_HPP
