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
    const float ms_sim = 1e16;
    const float kp_sim = 46400.;
public:
    EvrardGlassSphereCooling(std::string initBlock):
    sphexa::EvrardGlassSphere<Dataset>(initBlock) {}
    cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, size_t cbrtNumPart,
                                                 Dataset& simData) const override
    {

        chemistry_data grackleData = simData.chem.cooling_data.getDefaultChemistryData();

        grackleData.use_grackle = 1;
        grackleData.with_radiative_cooling = 1;
        grackleData.primordial_chemistry = 0;
        grackleData.dust_chemistry = 0;
        grackleData.metal_cooling = 0;
        grackleData.UVbackground = 0;

        simData.chem.cooling_data.init(ms_sim,
                                       kp_sim,
                                       0,
                                       grackleData,
                                       std::nullopt,
                                       std::nullopt);

        auto box = sphexa::EvrardGlassSphere<Dataset>::init(rank, numRanks, cbrtNumPart, simData);
        cooling::initGrackleData(simData.chem, simData.hydro.x.size());
        return box;
    }
};


#endif //SPHEXA_EVRARD_COOLING_INIT_HPP
